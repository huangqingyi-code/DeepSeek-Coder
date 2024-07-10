import time
import os
import json
import torch
from human_eval.evaluation import evaluate_functional_correctness
from transformers import AutoTokenizer
from utils.dataset import HumanEvalDataset
from utils.utils import cleanup_code
import numpy as np
import torch
import torch.nn.functional as F
import json
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from pathlib import Path
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm


class HumanEval:
    """
    HumanEval evaluation class.
    """

    def __init__(
        self,
        data_root,
        language="python",
        log_dir=None,
        issft=False,
        inference_increment=True,
        n_sample=1,
        k_sample=1,
    ):
        self.data_root = data_root
        self.k = k_sample
        self.n_sample = n_sample
        self.language = language
        self.log_dir = log_dir
        self.sft = issft
        self.inference_increment = inference_increment
        os.makedirs(self.log_dir, exist_ok=True)

    @torch.no_grad()
    def eval_model(self, args):
        """
        Evaluate the model on HumanEval.
        """
        assert (
            self.log_dir is not None
        ), "log_dir should not be None when evaluating humaneval"
        dataset = HumanEvalDataset(
            self.data_root,
            sample_num=self.n_sample,
            language=self.language,
            issft=self.sft,
        )
        model_name_or_path = args.model
        print("model", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(
            "load tokenizer {} from {} over.".format(
                tokenizer.__class__, model_name_or_path
            )
        )

        llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=1,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
        )
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            top_p=0.95,
            stop_token_ids=[tokenizer.eos_token_id],
        )
        messages_list = []
        for i in range(len(dataset)):
            data = dataset[i]
            prompt = data["prompt"].strip()
            messages_list.append(prompt)
        outputs = llm.generate(messages_list, sampling_params=sampling_params)
        assert len(dataset) == len(outputs), "dataset and outputs different lengths."
        log_file = os.path.join(self.log_dir, f"{self.language}.json")
        tmpfile = open(log_file, "w")
        for i, output in enumerate(tqdm(outputs)):
            data = dataset[i]
            output = output.outputs[0].text
            output = cleanup_code(
                output,
                self.language,
                "humaneval",
                self.sft,
                dataset.stopwords,
            )
            # sft mode does not need original prompt
            if not self.sft:
                suffixprediction = data["original_prompt"] + "\n" + output
            res = {
                "task_id": data["task_id"],
                "generation": suffixprediction,
                "prompt": data["original_prompt"],
            }
            tmpfile.write(json.dumps(res) + "\n")

        tmpfile.close()
        # calculate the final score of pass@k
        self._calculate_final_score(log_file)
        return

    def _calculate_final_score(self, logfilepath):
        """
        Calculate the final score.
        """
        res = evaluate_functional_correctness(
            input_file=logfilepath,
            problem_file=os.path.join(
                self.data_root, f"humaneval-{self.language}.jsonl"
            ),
            tmp_dir=self.log_dir,
            language=self.language,
        )
        print("score is", res["pass@%d" % self.k])
        os.remove(logfilepath)
        return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument(
        "--model",
        type=str,
        help="model name or path",
        default="/data0/pretrained-models/qwen2-7b",
    )

    parser.add_argument("--language", type=str, default="python")
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/home/qyhuang/DeepSeek-Coder/Evaluation/HumanEval/data",
    )
    args = parser.parse_args()

    logdir = args.logdir
    language = args.language

    if logdir == "":
        logdir = "tmp/"

    evaluator = HumanEval(
        data_root=args.dataroot,
        log_dir=logdir,
        n_sample=1,
        language=language,
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluator.eval_model(args)
