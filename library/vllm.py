import copy
from typing import Any

import numpy as np
import pygtrie
import regex
import torch
from transformers import (
    AutoConfig,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    MaxLengthCriteria,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)
from vllm import LLM, SamplingParams

from ._find import Find
from ._gen import Gen
from ._select import Select
from .backend import PathFinder
from .templates import LLAMA_CHAT_TEMPLATE
from .trie import MarisaTrie, Trie


def can_be_int(s):
    try:
        int(s)  # Try converting `s` to int
        return True
    except ValueError:
        return False  # Return False if a ValueError is raised


class ModelVLLMBackend(PathFinder):
    def __init__(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizer,
        trust_remote_code: bool = False,
        template: str = None,
        seed: int = 42,
    ) -> None:
        super().__init__(model_name)
        num_gpus = torch.cuda.device_count()
        self.model = LLM(
            model_name,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=num_gpus,
            seed=seed,
        )
        self.template = template
        self.tokenizer = tokenizer
        self.trust_remote_code = trust_remote_code

        self.temperature = 0.0
        self.top_p = 1.0
        self.max_tokens = 1000

    def _current_prompt(self):
        if isinstance(self.chat, list):
            prompt_render = self.tokenizer.apply_chat_template(
                self.chat,
                tokenize=False,
                add_generation_prompt=self.chat[-1]["role"] != "assistant",
                chat_template=self.template,
            )
        else:
            prompt_render = self.chat
        return prompt_render

    def _consume_assistant_text(self, value):
        self.prefix_text += value
        match = regex.match(
            r"(.*?)" + regex.escape(value) + r"(.*?)",
            self.text_to_consume,
            regex.DOTALL,
        )
        if match:
            self.text_to_consume = self.text_to_consume[len(match.group()) :]
            self.prefix_text = ""
        else:
            self.text_to_consume = ""

    def _get_gen(self, value: Gen):
        self.temperature = value.temperature
        self.top_p = value.top_p
        self.max_tokens = value.max_tokens
        if value.stop_regex is None:
            r = r"(.*?)"
        else:
            r = rf"(.*?)({value.stop_regex})"

        return self.run(self, r, value.name, True, value.save_stop_text)

    def _get_find(self, value: Find):
        self.temperature = value.temperature
        self.top_p = value.top_p
        return self.run_find(self, value.regex, value.name)

    def _get_select(self, value: Select):
        if all(can_be_int(x) for x in value.options):
            r = r"(\d+)"
        else:
            r = r"("
            r += r"|".join([regex.escape(o) for o in value.options])
            r += r")"
        return self.run(self, r, value.name, False, False)

    def run_find(self, lm, r, name):
        if lm.text_to_consume == "":
            lm.text_to_consume = lm.request()
        original_res = lm.text_to_consume
        match = regex.search(r, lm.text_to_consume)
        if match:
            res = match.group(0)
            lm._variables[name] = res
            return res, original_res
        else:
            raise Exception(f"Regex {r} not found in {lm.text_to_consume}")

    def run(self, lm, r, name, is_gen, save_stop_text):
        if lm.text_to_consume == "":
            lm.text_to_consume = lm.request()

        if regex.search(r, lm.text_to_consume):
            match = regex.match(r + r"(.*?)", lm.text_to_consume, regex.DOTALL)
            if match:
                # complete match
                match_res = match.group()
                if save_stop_text:
                    res = match.group()
                    lm.text_to_consume = lm.text_to_consume[len(match_res) :]
                else:
                    res = match.group(1)
                    lm.text_to_consume = lm.text_to_consume[len(match.group(1)) :]
            else:
                match = regex.findall(r, lm.text_to_consume, regex.DOTALL)[0]
                lm.text_to_consume = ""  # reset since this was a search of the response
                res = match
        elif is_gen:
            # not stop token
            res = lm.text_to_consume
            lm.text_to_consume = ""
        else:
            raise Exception(f"Cant find {r} in {lm.text_to_consume}")
        return res

    def _format_prompt(self):
        if isinstance(self.chat, list):
            tmp_chat = (
                self.chat[:-1]
                if self.chat[-1]["role"] == "assistant"
                and self.chat[-1]["content"] == ""
                else self.chat
            )  # prevent empty assistant block to be passed to the tokenizer

            prompt_render = self.tokenizer.apply_chat_template(
                tmp_chat,
                tokenize=False,
                add_generation_prompt=(
                    self.chat[-1]["content"] == ""
                ),  # we are already in assistant block, need to check if it is empty
                chat_template=self.template,
            )
        else:
            prompt_render = self.chat

        input_ids = self.tokenizer(
            prompt_render, return_tensors="np", add_special_tokens=True
        ).input_ids.tolist()
        return prompt_render, input_ids

    def request(self):
        prompt_render, input_ids = self._format_prompt()
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        output = self.model.generate(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        res = output[0].outputs[0].text
        return res
