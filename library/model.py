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


class Model:
    open_block = None
    empty_block = True

    token_in = 0
    token_out = 0

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

    def _format_chat_entry_as_html(self, entry):
        # Bold the role tag
        role_tag = f'<strong>{entry["role"].upper()}</strong>'
        return f'<div>{role_tag}: {entry["content"]}</div>'

    def html(self):
        if isinstance(self.chat, list):
            # Process each chat entry and format it as HTML
            html_entries = [
                self._format_chat_entry_as_html(entry) for entry in self.chat
            ]
            prompt_render = "".join(html_entries)
        else:
            # Format a single chat entry as HTML
            prompt_render = self._format_chat_entry_as_html(self.chat)
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
                        else {
                            "do_sample": False,
                            "temperature": 1.0,
                            "top_p": 1.0,
                        }
                    ),
                )
                model_config.update(generation_config.to_dict())

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
                if res.endswith(self.tokenizer.eos_token):
                    res = res[: -len(self.tokenizer.eos_token)]
                # remove end pattern if it exists and save_stop_text is True
                if not value.save_stop_text and value.stop_regex is not None:
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
                lm.token_in = len(input_ids[0])
                lm.token_out = len(output[0]) - len(input_ids[0])
                original_res = res
            elif isinstance(value, Find):
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
                        else {
                            "do_sample": False,
                            "temperature": 1.0,
                            "top_p": 1.0,
                        }
                    ),
                )
                model_config.update(generation_config.to_dict())

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
                if res.endswith(self.tokenizer.eos_token):
                    res = res[: -len(self.tokenizer.eos_token)]
                # remove end pattern if it exists and save_stop_text is True
                original_res = res
                lm._variables[f"PATHFINDER_ORIGINAL_{value.name}"] = res
                match = regex.search(value.regex, res)
                if match:
                    res = match.group(0)
                else:
                    raise Exception(f"Regex {value.regex} not found in {original_res}")
                lm.token_in = len(input_ids[0])
                lm.token_out = len(output[0]) - len(input_ids[0])
            elif isinstance(value, Select):
                model_config = AutoConfig.from_pretrained(self.model.name_or_path)
                generation_config = GenerationConfig(
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    renormalize_logits=True,
                    pad_token_id=(
                        model_config.pad_token_id
                        if model_config.pad_token_id
                        else model_config.eos_token_id
                    ),
                )
                model_config.update(generation_config.to_dict())

                options_text = [
                    self.tokenizer.decode(
                        self.tokenizer.encode(
                            prompt_render + option, add_special_tokens=False
                        ),
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
                    logprobs_result = gen_obj.scores[0][0].cpu().numpy()

                    top_logprobs = np.argsort(-logprobs_result)

                    for i, token in enumerate(top_logprobs):
                        # check if the token is in the trie
                        current_prefix = prefix + (token,)
                        current_prefix_decoded = self.tokenizer.decode(
                            current_prefix, skip_special_tokens=False
                        )
                        try:
                            extension_options = token_map.items(
                                prefix=current_prefix_decoded
                            )
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

                res = self.tokenizer.decode(
                    prefix[prompt_length:], skip_special_tokens=False
                )
                lm.token_in = prompt_length
                lm.token_out = len(prefix) - prompt_length
                if res.endswith(self.tokenizer.eos_token):
                    res = res[: -len(self.tokenizer.eos_token)]
                original_res = res
            else:
                raise Exception("Invalid state")
            # Save the result
            lm._variables[value.name] = res
            if isinstance(lm.chat, list):
                lm.chat[-1]["content"] += original_res
            else:
                lm.chat += original_res

        return lm

    def __getitem__(self, key):
        if key in self._variables:
            return self._variables[key]

    def set(self, key, value):
        lm = self.copy()
        lm._variables[key] = value
        return lm
