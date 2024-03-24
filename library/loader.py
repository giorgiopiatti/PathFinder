from auto_gptq import exllama_set_max_input_length
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def get_model(name, is_api=False):
    if is_api:
        return ModelAPI(name)
    trust_remote_code = False
    use_fast = True
    extend_context_length = True
    load_bettertransformer = True
    if "metamath" in name.lower():
        cls = MetaMath
    elif "llama" in name.lower():
        cls = LlamaChat
    # elif "yi" in name.lower():
    #     cls = YiChat
    #     trust_remote_code = True
    #     use_fast = False
    #     extend_context_length = False
    elif "smaug" in name.lower():
        cls = LlamaChat
    elif "vicuna" in name.lower():
        cls = Vicuna
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

    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        revision=branch,
    )

    if load_bettertransformer:
        # NOTE solves this warning:
        # The current implementation of Falcon calls `torch.scaled_dot_product_attention` directly, this will be deprecated in the future
        # in favor of the `BetterTransformer` API. Please install the latest optimum library with `pip install -U optimum` and
        # call `model.to_bettertransformer()` to benefit from `torch.scaled_dot_product_attention` and future performance optimizations.
        # https://huggingface.co/docs/optimum/bettertransformer/overview
        model = BetterTransformer.transform(model, keep_original_model=False)

    if "gptq" in name.lower() and extend_context_length:
        model = exllama_set_max_input_length(model, max_input_length=4096)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=use_fast)
    return cls(model=model, tokenizer=tokenizer)
