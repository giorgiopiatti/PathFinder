import copy
from typing import Any

import numpy as np
import pygtrie
import regex
import torch
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
    generate_enforced,
)
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

                options_tokens = [
                    self.tokenizer.encode(
                        prompt_render + option + self.tokenizer.eos_token,
                        add_special_tokens=False,
                    )
                    for option in value.options
                ]
                # build a trie of the options
                token_map = pygtrie.Trie()
                for i, option in enumerate(options_tokens):
                    token_map[option] = i

                def recursive_select(current_prefix, allow_token_extension=True):
                    """This returns a dictionary of scores for each option (keyed by the option index)."""

                    # find which select options are possible
                    try:
                        extension_options = token_map.items(prefix=current_prefix)
                    except KeyError:
                        return {}

                    # this is the dictionary of logprobs for each option we will return
                    # note that the logprobs are just for this branch point and below in the decision tree
                    logprobs_out = {option[0]: -1000 for option in extension_options}

                    # extend the prefix with the longest common prefix among the valid options
                    # we also stop early if we have one option
                    if len(extension_options) == 1:
                        logprobs_out[extension_options[0][0]] = (
                            0  # probability of 1.0 that we will select the only valid option
                        )
                        return logprobs_out
                    else:
                        match_index = len(current_prefix)
                        for i in range(
                            len(current_prefix),
                            min([len(o[0]) for o in extension_options]),
                        ):
                            if len(set([o[0][i] for o in extension_options])) > 1:
                                break
                            match_index += 1
                        if match_index > len(current_prefix):
                            current_prefix += extension_options[0][0][
                                len(current_prefix) : match_index
                            ]
                            # extension_options = [(option[i:], index) for option,index in extension_options]

                    # bias the logits towards valid options
                    logit_bias = {}
                    for option_tokens, index in extension_options:
                        logit_bias[option_tokens[match_index]] = 100

                    # check for where we are at the end of the prefix
                    if len(logit_bias) == 0 and current_prefix in [
                        o[0] for o in extension_options
                    ]:
                        logprobs_out[current_prefix] = 0
                        return logprobs_out

                    # generate the token logprobs
                    gen_obj = self.model.generate(
                        inputs=input_ids,
                        generation_config=generation_config,
                        logits_processor=LogitsProcessorList(
                            [
                                BiasLogitsProcessor(
                                    self.model, self.tokenizer.vocab_size, logit_bias
                                )
                            ]
                        ),
                    )

                    logprobs_result = gen_obj.scores[0][0]

                    # convert the logprobs keys from string back to token ids if needed
                    top_logprobs = {}
                    for k, v in enumerate(logprobs_result):
                        top_logprobs[k] = k

                    # no need to explore all branches if we are just taking the greedy max
                    # if logprobs is None:
                    #     max_key = max(top_logprobs, key=top_logprobs.get)
                    #     top_logprobs = {max_key: top_logprobs[max_key]}

                    # for each possible next token, see if it grows the prefix in a valid way
                    for token, logprob in top_logprobs.items():
                        sub_logprobs = recursive_select(current_prefix + [token])

                        # we add the logprob of this token to the logprob of the suffix
                        for k in sub_logprobs:

                            # p1 = np.exp(logprobs_out[k])
                            # p2 = np.exp(sub_logprobs[k] + logprob)
                            # or_prob = p1 + p2 - p1 * p2
                            # logprobs_out[k] = np.log(or_prob)

                            # New Code Using Log-Sum-Exp
                            a = logprobs_out[k]
                            b = sub_logprobs[k] + logprob

                            # Computing log(exp(a) + exp(b)) in a stable way
                            logprobs_out[k] = np.logaddexp(a, b)

                            # logprobs_out[k] = np.log(
                            #     np.exp(logprobs_out[k]) - np.exp(a + b)
                            # )
                            x = logprobs_out[k]
                            if x > (a + b):
                                logprobs_out[k] = x + np.log1p(-np.exp(a + b - x))
                            else:
                                logprobs_out[k] = -np.inf

                            # logprobs_out[k] = np.log1p(-np.exp(a + b - logprobs_out[k]))

                    return logprobs_out

                # recursively compute the logprobs for each option
                option_logprobs = recursive_select([])

                # convert the key from a token list to a string
                coded_prompt = self.tokenizer.decode(input_ids[0])
                option_logprobs = {
                    self.tokenizer.decode(k)[len(coded_prompt) :]: v
                    for k, v in option_logprobs.items()
                }

                # select the option with the highest logprob
                res = max(option_logprobs, key=option_logprobs.get)
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
