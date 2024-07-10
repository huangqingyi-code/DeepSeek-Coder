import argparse
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm
import transformers
from utils.utils import extract_generation_code, languge_settings
from transformers import AutoTokenizer
from human_eval.evaluation import evaluate_functional_correctness
from vllm import LLM, SamplingParams

data_abs_dir = Path(__file__).parent / "data"


def build_deepseekcoder_instruction(languge: str, question: str):
    return """
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
""".strip().format(
        languge.lower(), question.strip()
    )


def generate_main(args):
    model_name_or_path = args.model_path
    lang = args.language
    saved_path = args.output_path
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")

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
        temperature=0,
        max_tokens=1024,
        top_p=0.95,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    examples = [json.loads(x) for x in open(problem_file) if x.strip()]
    print("Read {} examples for evaluation over.".format(len(examples)))
    messages_list = []
    for example in tqdm(examples, desc="Generating"):
        prompt = build_deepseekcoder_instruction(
            languge_settings[lang]["full_name"], example["prompt"]
        )
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
        example["output"] = output
        example = extract_generation_code(example, lang_code=lang)
        generated_examples.append(example)

    print("Generate all over!!!")
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
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang,
    )
    print(lang, result, model_name_or_path)
    pass


def evaluation_only(args):
    lang = args.language
    temp_dir = args.temp_dir
    assert os.path.exists(args.output_path), "Not fond output file: {}".format(
        args.output_path
    )
    os.makedirs(temp_dir, exist_ok=True)

    output_name = os.path.basename(args.output_path)
    output_examples = [json.loads(x) for x in open(args.output_path) if x.strip()]

    processed_examples = [
        extract_generation_code(ex, lang) for ex in tqdm(output_examples, "Processing")
    ]
    processed_path = os.path.join(temp_dir, output_name)
    with open(processed_path, "w", encoding="utf-8") as fw:
        for ex in processed_examples:
            fw.write(json.dumps(ex) + "\n")
        print(
            "Save {} processed examples into {} over!".format(
                len(processed_examples), processed_path
            )
        )

    problem_file = os.path.join(data_abs_dir, f"humaneval-{lang}.jsonl")
    from human_eval.evaluation import evaluate_functional_correctness

    result = evaluate_functional_correctness(
        input_file=processed_path,
        tmp_dir=temp_dir,
        n_workers=8,
        timeout=3.0,
        problem_file=problem_file,
        language=lang,
    )
    print(lang, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="model name or path",
        default="/data4/sft_output/qwen2-instruct-0709/checkpoint-1400",
    )
    parser.add_argument(
        "--gpus_num", type=int, default=1, help="the number of GPUs you want to use."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="output path of your generation",
        default="outputs/qwen2-sft.json",
    )
    parser.add_argument("--language", type=str, help="langauge", default="python")
    parser.add_argument(
        "--temp_dir", type=str, help="temp dir for evaluation", default="tmp"
    )
    parser.add_argument("--seed", type=int, help="seed", default=42)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.set_seed(args.seed)
    generate_main(args)
