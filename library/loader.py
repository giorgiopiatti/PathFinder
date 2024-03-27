import torch
from auto_gptq import exllama_set_max_input_length
from optimum.bettertransformer import BetterTransformer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .api import ModelAPI
from .chat import (
    ChatML,
    DeepSeek,
    LlamaChat,
    MetaMath,
    MistralInstruct,
    MixtralInstruct,
    Model,
    Vicuna,
)


def get_model(name, is_api=False, seed=42):
    if is_api:
        return ModelAPI(name, seed)
    trust_remote_code = False
    use_fast = True
    extend_context_length = True
    if "metamath" in name.lower():
        cls = MetaMath
    elif "llama" in name.lower():
        cls = LlamaChat
    elif "smaug" in name.lower():
        cls = LlamaChat
    elif "vicuna" in name.lower():
        cls = Vicuna
    elif "pro-mistral" in name.lower():
        cls = ChatML
    elif "dbrx-instruct" in name.lower():
        trust_remote_code = True
        cls = ChatML
    elif "mistral" in name.lower() and "instruct" in name.lower():
        cls = MistralInstruct
    elif "mixtral" in name.lower() and "instruct" in name.lower():
        cls = MixtralInstruct
    elif "hermes-2-mixtral" in name.lower() or "qwen" in name.lower():
        cls = ChatML
    elif "deepseek" in name.lower():
        cls = DeepSeek
    else:
        raise ValueError(f"Unknown model name {name}")

    branch = "main"
    if "gptq" in name.lower():
        branch = "gptq-4bit-32g-actorder_True"

        if "metamath" in name.lower() and not "mistral" in name.lower():
            branch = "gptq-4-32g-actorder_True"
        elif "smaug" in name.lower():
            branch = "main"
        elif name.startswith("Qwen"):
            branch = "main"
            extend_context_length = False

    model_config = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        revision=branch,
        # attn_implementation=(
        #     "flash_attention_2" if not "gptq" in name.lower() else None
        # ),
        torch_dtype=model_config.torch_dtype if not "gptq" in name.lower() else None,
    )

    if "gptq" in name.lower() and extend_context_length:
        model = exllama_set_max_input_length(model, max_input_length=4096)

    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
    return cls(model=model, tokenizer=tokenizer)
