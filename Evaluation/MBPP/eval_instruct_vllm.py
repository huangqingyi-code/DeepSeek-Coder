import argparse
import json
import os
import transformers
import re
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
import shutil
data_abs_dir = Path(__file__).parent / "data"

from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.evaluation import evaluate_functional_correctness


def create_dir(output_dir):
    if os.path.exists(output_dir):
        if not os.access(output_dir, os.W_OK):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            os.chmod(output_dir, 0o777)
            print("not write permission, makedir:", output_dir)
        else:
            print(f"{output_dir} exists!")
    else:
        os.makedirs(output_dir)
        os.chmod(output_dir, 0o777)
        print("makedir:", output_dir)


def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str = None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(
            q.strip(), "\n".join(tests)
        )
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex["text"], ex["test_list"], ex["code"]
        ex_prompt = format_test_example(q, test, code)
        example_prompt = "- Example {}:\n{}".format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex["text"], ex["test_list"], ex["code"]

        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = """
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
""".strip().format(
            "\n\n".join(examples_str), prompt
        )
        yield {"task_id": ex["task_id"], "prompt": prompt_with_shots}


def convert_for_evaluation(example):
    gpt_completion = example["gpt_completion"]
    generation = gpt_completion
    try:
        code_block: str = re.findall(
            f"```python\n(.*?)```", gpt_completion, re.DOTALL | re.IGNORECASE
        )[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    example["generation"] = generation
    return example


def generate_main(args):
    model_name_or_path = args.model_path
    temp_dir = args.temp_dir
    create_dir(temp_dir)
    # os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"mbpp.jsonl")

    print("model", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print(
        "load tokenizer {} from {} over.".format(
            tokenizer.__class__, model_name_or_path
        )
    )

    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=args.gpus_num,
        max_model_len=8192,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0, max_tokens=1024, stop_token_ids=[tokenizer.eos_token_id]
    )

    examples = list(read_test_examples(problem_file))
    print("Read {} examples for evaluation over.".format(len(examples)))
    messages_list = []
    for example in tqdm(examples, desc="Generating"):
        prompt = example["prompt"]
        message = [{"role": "user", "content": prompt}]
        messages_list.append(
            tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
        )

    outputs = llm.generate(messages_list, sampling_params=sampling_params)
    generated_examples = []
    for i, output in enumerate(tqdm(outputs)):
        output = output.outputs[0].text
        example = examples[i]
        example["gpt_completion"] = output
        example = convert_for_evaluation(example)
        generated_examples.append(example)

    print("Generate all over!!!")
    # os.makedirs(args.save_dir, exist_ok=True)
    create_dir(args.save_dir)
    saved_path = os.path.join(args.save_dir, "results_mbpp.json")
    with open(saved_path, "w", encoding="utf-8") as fw:
        for ex in generated_examples:
            fw.write(json.dumps(ex) + "\n")
        print(
            "Save {} processed examples into {} over!".format(
                len(generated_examples), saved_path
            )
        )

    result = evaluate_functional_correctness(
        input_file=saved_path,
        tmp_dir=temp_dir,
        problem_file=os.path.join(data_abs_dir, f"mbpp_test.jsonl"),
        language="python",
        is_mbpp=True,
    )
    print(result, model_name_or_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="model name or path",
        default="/data4/sft_output/qwen2-instruct-0709/checkpoint-1200",
    )
    parser.add_argument(
        "--gpus_num", type=int, default=1, help="the number of GPUs you want to use."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="output path of your generation",
        default="outputs",
    )
    parser.add_argument(
        "--temp_dir", type=str, help="temp dir for evaluation", default="tmp"
    )
    parser.add_argument("--seed", type=int, help="seed", default=42)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.set_seed(args.seed)
    generate_main(args)
