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

from ._find import Find
from ._gen import Gen
from ._select import Select
from .templates import LLAMA_CHAT_TEMPLATE
from .trie import MarisaTrie, Trie


class RegexStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_pattern, decode, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_regex = [regex.compile(stop_pattern)]
        else:
            self.stop_regex = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.decode = decode
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):
        # Only look at the generated part
        current_string = self.decode(
            input_ids[0][self.prefix_length :], skip_special_tokens=False
        )

        for s in self.stop_regex:
            if s.search(current_string):
                return True
        return False


class BiasLogitsProcessor(LogitsProcessor):
    """Simple token biasing."""

    def __init__(self, model, vocab_size, logit_bias):
        """Build a new BiasLogitsProcessor."""
        import torch

        self.bias_vector = torch.zeros(vocab_size)
        for token, bias in logit_bias.items():
            self.bias_vector[token] = bias
        self.bias_vector = self.bias_vector.to(model.device)

    def __call__(self, input_ids, scores):
        return scores + self.bias_vector


from .backend import PathFinder


class Model(PathFinder):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        trust_remote_code: bool = False,
        template: Any = None,
    ) -> None:
        super().__init__(model.name_or_path)
        self.model = model
        self.template = template
        self.tokenizer = tokenizer

        self.trust_remote_code = trust_remote_code

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
            prompt_render, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(self.model.device)
        return prompt_render, input_ids

    def _get_gen(self, value: Gen):
        prompt_render, input_ids = self._format_prompt()

        generation_config = GenerationConfig.from_pretrained(
            self.model.name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = (
                eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            )
        generation_config.update(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=value.max_tokens,
            **(
                {
                    "temperature": value.temperature,
                    "do_sample": True,
                    "top_p": value.top_p,
                }
                if value.temperature != 0.0
                else {
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                }
            ),
        )

        output = self.model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            stopping_criteria=(
                StoppingCriteriaList(
                    [
                        RegexStoppingCriteria(
                            value.stop_regex,
                            self.tokenizer.decode,
                            input_ids.shape[1],
                        )
                    ]
                )
                if value.stop_regex
                else None
            ),
        )

        res = self.tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=False
        )
        # use generation_config.eos_token_id, can be a list or a single value, detokenize and remove eos token
        if isinstance(eos_token_id, list):
            for eos_id in eos_token_id:
                eos = self.tokenizer.decode(eos_id)
                if res.endswith(eos):
                    res = res[: -len(eos)]
                    break
        else:
            eos = self.tokenizer.decode(eos_token_id)
            if res.endswith(eos):
                res = res[: -len(self.tokenizer.eos_token)]
        if not value.save_stop_text and value.stop_regex is not None:
            if isinstance(value.stop_regex, str):
                stop_regex = [regex.compile(value.stop_regex)]
            else:
                stop_regex = [regex.compile(pattern) for pattern in value.stop_regex]

            for p in stop_regex:
                if p.search(res):
                    res = p.sub("", res)
                    break
        self.token_in = len(input_ids[0])
        self.token_out = len(output[0]) - len(input_ids[0])

        return res

    def _get_find(self, value: Find):
        prompt_render, input_ids = self._format_prompt()
        generation_config = GenerationConfig.from_pretrained(
            self.model.name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = (
                eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            )
        generation_config.update(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=value.max_tokens,
            **(
                {
                    "temperature": value.temperature,
                    "do_sample": True,
                    "top_p": value.top_p,
                }
                if value.temperature != 0.0
                else {
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                }
            ),
        )
        output = self.model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            stopping_criteria=(
                StoppingCriteriaList(
                    [
                        RegexStoppingCriteria(
                            value.stop_regex,
                            self.tokenizer.decode,
                            input_ids.shape[1],
                        )
                    ]
                )
                if value.stop_regex
                else None
            ),
        )

        res = self.tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=False
        )

        # use generation_config.eos_token_id, can be a list or a single value, detokenize and remove eos token
        if isinstance(eos_token_id, list):
            for eos_id in eos_token_id:
                eos = self.tokenizer.decode(eos_id)
                if res.endswith(eos):
                    res = res[: -len(eos)]
                    break
        else:
            eos = self.tokenizer.decode(eos_token_id)
            if res.endswith(eos):
                res = res[: -len(self.tokenizer.eos_token)]
        # remove end pattern if it exists and save_stop_text is True
        original_res = res
        self._variables[f"PATHFINDER_ORIGINAL_{value.name}"] = res
        match = regex.search(value.regex, res)
        if match:
            res = match.group(0)
        else:
            raise Exception(f"Regex {value.regex} not found in {original_res}")
        self.token_in = len(input_ids[0])
        self.token_out = len(output[0]) - len(input_ids[0])
        return res, original_res

    def _get_select(self, value: Select):
        prompt_render, input_ids = self._format_prompt()
        model_config = AutoConfig.from_pretrained(
            self.model.name_or_path, trust_remote_code=self.trust_remote_code
        )
        pad_token_id = model_config.pad_token_id
        eos_token_id = model_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generation_config = GenerationConfig(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            renormalize_logits=True,
        )

        options_text = [
            self.tokenizer.decode(
                self.tokenizer.encode(prompt_render + option, add_special_tokens=True),
                skip_special_tokens=False,
            )
            for option in value.options
        ]
        # build a trie of the options
        token_map = pygtrie.Trie()
        for i, option in enumerate(options_text):
            token_map[option] = option

        # hack to deal with sentencepiece "" empty
        prefix = input_ids[0].tolist()
        if self.tokenizer.decode(prefix[-1]) == "":
            prefix = prefix[:-1]
        prefix = tuple(prefix)
        prompt_length = len(prefix)
        full_match = False
        need_more_tokens = True
        while need_more_tokens:
            # generate the token logprobs
            gen_obj = self.model.generate(
                inputs=torch.tensor([prefix], device=self.model.device),
                generation_config=generation_config,
            )
            logprobs_result = gen_obj.scores[0][0].to(dtype=torch.float32).cpu().numpy()

            top_logprobs = np.argsort(-logprobs_result)

            for i, token in enumerate(top_logprobs):
                # check if the token is in the trie
                current_prefix = prefix + (token,)
                current_prefix_decoded = self.tokenizer.decode(
                    current_prefix, skip_special_tokens=False
                )
                try:
                    extension_options = token_map.items(prefix=current_prefix_decoded)
                    partial_match = True
                except KeyError:
                    partial_match = False
                if partial_match:
                    prefix = current_prefix

                    for e in extension_options:
                        if e[1] == current_prefix_decoded:
                            # we have a full match
                            full_match = True
                            if len(extension_options) == 1:
                                # we have a unique match
                                need_more_tokens = False
                            break
                    break
                else:
                    # we have not found a partial match
                    if i == 0 and full_match:
                        # we had a full match before, so we are done
                        need_more_tokens = False
                        break

        res = self.tokenizer.decode(prefix[prompt_length:], skip_special_tokens=False)
        self.token_in = prompt_length
        self.token_out = len(prefix) - prompt_length
        if res.endswith(self.tokenizer.eos_token):
            res = res[: -len(self.tokenizer.eos_token)]
        return res
