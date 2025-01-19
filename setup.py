from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="pathfinder",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    license="MIT",
    description="A package for LLM-querying and output parsing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "accelerate",
        "transformers",
        "optimum",
        "einops",
        "tiktoken",  # for databricks models
        "hf_transfer",  # fast transfer of models
        # vllm
        # APIs
        "backoff",
        "openai",
        "mistralai",
        "anthropic",
        # path-finder specific
        "pygtrie",
        "numpy",
    ],
    url="https://github.com/giorgiopiatti/pathfinder",
    author="Giorgio Piatti",
)
