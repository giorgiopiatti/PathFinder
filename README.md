# PathFinder

The `pathfinder` library is a prompting library, that
wraps around the most common LLM inference backends (OpenAI, Azure OpenAI, Anthropic, Mistral, OpenRouter, `transformers` library and `vllm`) and allows for easy inference with LLMs.

## Development Notes
I'm building this library to be as modular as possible, so that it can be easily extended to support new models and new backends. Currently the supported models are hardcode in `pathfinder/library/loader.py`, but I'm planning to remove this and make the library more dynamic.

This is a *work in progress*, so expect breaking changes in the way a model is initialized. Nevertheless, this library helped me a lot in my research, especially  in [GovSim (Governance of the Commons Simulation)](https://github.com/giorgiopiatti/govsim), so I'm sharing it as separate project in the hope that it can help others as well.


## Supported LLMs
In principle, any LLM model can be used. We support the following families of models:

*APIs:*
- OpenAI: `gpt-4-turbo-2024-04-09`, `gpt-3.5-turbo-0125`, `gpt-4o-2024-05-13`
- Anthropic: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- Mistral (API):


*Open-weights models:*
- Mistral: `mistralai/Mistral-7B-Instruct-v0.2`, `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Llama-2: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`, `meta-llama/Llama-2-70b-chat-hf`
- Llama-3: `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B-Instruct`
- Qwen-1.5: `Qwen/Qwen1.5-72B-Chat-GPTQ-Int4`, `Qwen/Qwen1.5-110B-Chat-GPTQ-Int4`
- Phi-3: `microsoft/Phi-3-medium-4k-instruct`
- Vicuna
- Cohere


## Documentation
More documentation & examples coming soon.