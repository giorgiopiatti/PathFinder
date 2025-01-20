import torch
from accelerate import infer_auto_device_map
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .api import AnthropicAPI, AzureOpenAIAPI, MistralAPI, OpenAIAPI, OpenRouter, DeepSeekAPI, HumanAPI
from .chat import (
    ChatML,
    Cohere,
    DeepSeek,
    Llama3Chat,
    LlamaChat,
    MetaMath,
    MistralInstruct,
    MixtralInstruct,
    Model,
    Phi3,
    Vicuna,
)
from .model import Model


def get_api_model(name, seed):
    if "z-gpt" in name.lower():
        return AzureOpenAIAPI(name, seed)
    if "gpt" in name.lower():
        return OpenAIAPI(name, seed)
    elif "mistral" in name.lower():
        return MistralAPI(name, seed)
    elif "claude" in name.lower():
        return AnthropicAPI(name, seed)
    elif "openrouter" in name.lower():
        name = name.replace("openrouter-", "")
        return OpenRouter(name, seed)
    elif "deepseek" in name.lower():
        return DeepSeekAPI(name, seed)
    elif "human" in name.lower():
        return HumanAPI(name, seed)
    else:
        raise ValueError(f"Unknown model name {name}")


def get_gpu_memory_no_reserve(gpu_list):
    """
    Detect the total memory on each GPU in the list without reserving extra overhead.
    Returns a dictionary mapping each GPU to its total memory.
    """
    max_memory = {}
    for gpu in gpu_list:
        total_memory = torch.cuda.get_device_properties(gpu).total_memory
        max_memory[gpu] = f"{int(total_memory // (1024 ** 3))}GiB"  # Use full memory
    return max_memory


def get_available_gpus():
    """
    Returns a list of all available GPUs in the format [0, 1, ...].
    """
    if not torch.cuda.is_available():
        return []  # No GPUs available
    return [i for i in range(torch.cuda.device_count())]


def get_model(
    name,
    is_api=False,
    seed=42,
    backend_name="transformers",
    gpu_list=None,
):
    if gpu_list is None:
        gpu_list = get_available_gpus()
    if is_api:
        return get_api_model(name, seed)
    trust_remote_code = False
    use_fast = True
    extend_context_length = True
    if "metamath" in name.lower():
        cls = MetaMath
    elif "llama-3" in name.lower():
        cls = Llama3Chat
    elif "llama" in name.lower():
        cls = LlamaChat
    elif "phi-3" in name.lower():
        cls = Phi3
        trust_remote_code = True
    elif "smaug" in name.lower():
        cls = LlamaChat
    elif "vicuna" in name.lower():
        cls = Vicuna
    elif "wizardlm" in name.lower():
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
    elif "mathstral" in name.lower():
        cls = MistralInstruct
    elif "hermes-2-mixtral" in name.lower() or "qwen" in name.lower():
        cls = ChatML
    elif "deepseek" in name.lower():
        cls = DeepSeek
        trust_remote_code = True
    elif "command" in name.lower():
        cls = Cohere
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

        if not "thebloke" in name.lower():
            branch = "main"
            extend_context_length = False

    tokenizer = AutoTokenizer.from_pretrained(
        name, use_fast=use_fast, trust_remote_code=trust_remote_code
    )

    if backend_name == "transformers":
        import os
        from dotenv import load_dotenv
        load_dotenv()
        access_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        model_config = AutoConfig.from_pretrained(
            name, trust_remote_code=trust_remote_code, use_auth_token=access_token,
        )

        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="balanced",
            max_memory=get_gpu_memory_no_reserve(gpu_list),
            trust_remote_code=trust_remote_code,
            revision=branch,
            # attn_implementation=(
            #     "flash_attention_2" if not "gptq" in name.lower() else None
            # ),
            torch_dtype=(
                model_config.torch_dtype if not "gptq" in name.lower() else None
            ),
        )

        if "gptq" in name.lower() and extend_context_length:
            from auto_gptq import exllama_set_max_input_length

            model = exllama_set_max_input_length(model, max_input_length=4096)

        model = torch.compile(model)
        backend = Model(
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            template=cls().template,
        )
        model.name_or_path = name
        print("Model device")
        print(model.device)

    elif backend_name == "vllm":
        from .vllm import ModelVLLMBackend

        backend = ModelVLLMBackend(
            name, tokenizer, trust_remote_code, cls().template, seed
        )

    return backend
