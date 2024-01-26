import copy
from typing import Any

import torch
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
    generate_enforced,
)
from transformers import (
    AutoConfig,
    GenerationConfig,
    MaxLengthCriteria,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

from ._gen import Gen
from ._select import Select
from .templates import LLAMA_CHAT_TEMPLATE

import regex


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


class Model:
    open_block = None
    empty_block = True

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        self.model = model
        self.template = None
        self.tokenizer = tokenizer
        # these are the state variables stored with the model
        self._variables = {}
        self._variables_log_probs = {}

        self.chat = []

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

    def __str__(self) -> str:
        pass

    def copy(self):
        """Create a shallow copy of the model object."""

        # start with a shallow copy
        new_lm = copy.copy(self)

        # then copy a few things we need deeper copies of
        new_lm._variables = self._variables.copy()
        new_lm._variables_log_probs = self._variables_log_probs.copy()

        if isinstance(self.chat, list):
            new_lm.chat = self.chat.copy()
        else:
            new_lm.chat = self.chat

        return new_lm

    def __add__(self, value):
        # we hav string, gen, select
        lm = self.copy()

        if len(lm.chat) == 0 and Model.empty_block and Model.open_block is None:
            # We are not in a chat block, so we simply add the string
            lm.chat = ""
        elif Model.open_block is not None and Model.open_block.init_tag:
            Model.open_block.init_tag = False
            lm.chat.append(
                {
                    "role": Model.open_block.role,
                    "content": "",
                }
            )

        if isinstance(value, str):
            if isinstance(lm.chat, list):
                lm.chat[-1]["content"] += value
            else:
                lm.chat += value
        else:
            if isinstance(lm.chat, list):
                if lm.chat[-1]["role"] != "assistant":
                    raise Exception(
                        f"{value} can be used only in assistant block, not in"
                        f" {lm.chat[-1]['role']} block!"
                    )
                prompt_render = self.tokenizer.apply_chat_template(
                    lm.chat,
                    tokenize=False,
                    add_generation_prompt=lm.chat[-1]["role"] != "assistant",
                    chat_template=self.template,
                )
            else:
                prompt_render = lm.chat

            # Tokenize
            input_ids = self.tokenizer(
                prompt_render, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)

            # Run specific generation
            if isinstance(value, Gen):
                model_config = AutoConfig.from_pretrained(self.model.name_or_path)

                generation_config = GenerationConfig(
                    max_new_tokens=value.max_tokens,
                    pad_token_id=(
                        model_config.pad_token_id
                        if model_config.pad_token_id
                        else model_config.eos_token_id
                    ),
                    **(
                        {
                            "temperature": value.temperature,
                            "do_sample": True,
                            "top_p": value.top_p,
                        }
                        if value.temperature != 0.0
                        else {}
                    ),
                )
                model_config.update(generation_config.to_dict())

                output = self.model.generate(
                    inputs=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList(
                        [
                            RegexStoppingCriteria(
                                value.stop_regex,
                                self.tokenizer.decode,
                                input_ids.shape[1],
                            )
                        ],
                    ),
                )

                res = self.tokenizer.decode(
                    output[0][input_ids.shape[1] :], skip_special_tokens=False
                )
                if res.endswith(self.tokenizer.eos_token):
                    res = res[: -len(self.tokenizer.eos_token)]
                # remove end pattern if it exists and save_stop_text is True
                if not value.save_stop_text:
                    if isinstance(value.stop_regex, str):
                        stop_regex = [regex.compile(value.stop_regex)]
                    else:
                        stop_regex = [
                            regex.compile(pattern) for pattern in value.stop_regex
                        ]

                    for p in stop_regex:
                        if p.search(res):
                            res = p.sub("", res)
                            break
            elif isinstance(value, Select):
                model_config = AutoConfig.from_pretrained(self.model.name_or_path)
                generation_config = GenerationConfig(
                    max_length=4096,
                    pad_token_id=(
                        model_config.pad_token_id
                        if model_config.pad_token_id
                        else model_config.eos_token_id
                    ),
                )
                model_config.update(generation_config.to_dict())

                regex_select = ""
                for i, o in enumerate(value.options):
                    if i == 0:
                        regex_select += o
                    else:
                        regex_select += "|" + o

                parser = RegexParser(regex_select)
                prefix_function = build_transformers_prefix_allowed_tokens_fn(
                    self.tokenizer, parser
                )

                output = self.model.generate(
                    inputs=input_ids,
                    generation_config=generation_config,
                    prefix_allowed_tokens_fn=prefix_function,
                )
                res = self.tokenizer.decode(
                    output[0][input_ids.shape[1] :], skip_special_tokens=False
                )
                if res.endswith(self.tokenizer.eos_token):
                    res = res[: -len(self.tokenizer.eos_token)]
            else:
                raise Exception("Invalid state")
            # Save the result
            lm._variables[value.name] = res
            if isinstance(lm.chat, list):
                lm.chat[-1]["content"] += res
            else:
                lm.chat += res

        return lm

    def __getitem__(self, key):
        if key in self._variables:
            return self._variables[key]

    def set(self, key, value):
        lm = self.copy()
        lm._variables[key] = value
        return lm
