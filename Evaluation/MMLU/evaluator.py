import inspect
import json
import os
from typing import Any, Dict, List, Optional
import argparse
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file
import transformers
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path
import shutil

# from ..data import get_template_and_fix_tokenizer

# from ..hparams import get_eval_args
# from ..model import load_model, load_tokenizer
from template import get_eval_template, CHOICES

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]


data_abs_dir = Path(__file__).parent / "data"


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


class Evaluator:
    def __init__(self, args) -> None:
        # self.model_args, self.data_args, self.eval_args, finetuning_args = (
        #     get_eval_args(args)
        # )
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.gpus_num,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            top_p=0.95,
            stop_token_ids=[self.tokenizer.eos_token_id],
            logprobs=20,
        )

        self.tokenizer.padding_side = (
            "right"  # avoid overflow issue in batched inference for llama2
        )

        # self.template = get_template_and_fix_tokenizer(
        #     self.tokenizer, self.data_args.template
        # )
        self.eval_template = get_eval_template(args.lang)
        self.choice_inputs = [
            self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES
        ]
        self.label_map = {}
        for label in ["A", "B", "C", "D"]:
            self.label_map[label] = self.tokenizer.convert_tokens_to_ids(label)

    def get_answer(self, output):
        import numpy as np

        candidate_logits = []
        for label in ["A", "B", "C", "D"]:
            try:
                candidate_logits.append(
                    output.outputs[0].logprobs[0][self.label_map[label]].logprob
                )
            except:
                # If an option is not in the first 1000, set its logit to -100
                # print(
                #     "Warning: {} not found. Artificially adding log prob of -100.".format(
                #         label
                #     )
                # )
                candidate_logits.append(-100)
        # 全是-100
        if len(set(candidate_logits)) == 1 and candidate_logits[0] == -100:
            print("Warning candidate_logits:", candidate_logits)
            return "N"
        candidate_logits = torch.tensor(candidate_logits)
        probs = torch.nn.functional.softmax(
            candidate_logits,
            dim=0,
        ).numpy()
        answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]
        return answer

    def eval(self) -> None:
        mapping = cached_file(
            path_or_repo_id=os.path.join(data_abs_dir, self.args.task),
            filename="mapping.json",
        )

        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)
        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            if (
                "trust_remote_code" in inspect.signature(load_dataset).parameters
            ):  # for datasets==2.16.0
                kwargs = {"trust_remote_code": True}
            else:
                kwargs = {}

            dataset = load_dataset(
                path=os.path.join(data_abs_dir, self.args.task),
                name=subject,
                **kwargs,
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(
                len(dataset[args.split]),
                desc="Formatting batches",
                position=1,
                leave=False,
            ):
                support_set = (
                    dataset["train"]
                    .shuffle()
                    .select(range(min(self.args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset[self.args.split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                )
                inputs.append(
                    self.tokenizer.apply_chat_template(
                        messages[0:-1], tokenize=False, add_generation_prompt=True
                    )
                )
                labels.append(messages[-1]["content"])

            llm_outputs = self.llm.generate(
                inputs, sampling_params=self.sampling_params
            )
            for output in llm_outputs:
                answer = self.get_answer(output)
                outputs.append(answer)

            corrects = np.array(outputs) == np.array(labels)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate(
                [category_corrects[category_name], corrects], axis=0
            )
            category_corrects["Average"] = np.concatenate(
                [category_corrects["Average"], corrects], axis=0
            )
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)

    def _save_results(
        self,
        category_corrects: Dict[str, np.ndarray],
        results: Dict[str, Dict[int, str]],
    ) -> None:
        score_info = "\n".join(
            [
                "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.args.save_dir is not None:
            # os.makedirs(self.args.save_dir, exist_ok=True)
            create_dir(self.args.save_dir)
            with open(
                os.path.join(self.args.save_dir, f"results_{args.task}.json"),
                "w",
                encoding="utf-8",
                newline="\n",
            ) as f:
                json.dump(results, f, indent=2)

            with open(
                os.path.join(self.args.save_dir, f"results_{args.task}.log"),
                "w",
                encoding="utf-8",
                newline="\n",
            ) as f:
                f.write(score_info)


def run_eval(args) -> None:
    evalutor = Evaluator(args)
    evalutor.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="model name or path",
        default="/data0/pretrained-models/Qwen2-7B-Instruct",
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
    parser.add_argument(
        "--lang", type=str, help="prompt langauge", default="zh", choices=["zh", "en"]
    )
    parser.add_argument(
        "--task",
        type=str,
        help="eval task",
        default="cmmlu",
        choices=["cmmlu", "mmlu", "ceval"],
    )
    parser.add_argument(
        "--split",
        type=str,
        help="eval data split",
        default="test",
        choices=["test", "validation"],
    )
    parser.add_argument("--n_shot", type=int, help="n shot", default=5)
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument("--save_dir", type=str, help="save dir", default="outputs")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    transformers.set_seed(args.seed)
    run_eval(args)
