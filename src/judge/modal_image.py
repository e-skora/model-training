"""The one Modal container image every GPU job in this repo uses.

Pins: Unsloth 2026.10.7 (newest on PyPI, 2026-10-22) and the versions uv picks
for it on Linux with Python 3.11. The older pins from Modal's Unsloth example
(Unsloth 2025.7.8, transformers 4.54.0) could not load Qwen3-4B-Instruct-2507
or Phi-4-mini-instruct. torch 2.14.1 brings CUDA 13.0; Modal's driver supports
up to CUDA 13.3 (https://modal.com/docs/guide/cuda).
Change pins here only, then rerun `scripts/hello_gpu.py` to prove the image still works.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.15.0",
        "bitsandbytes==0.50.2",
        "datasets==4.8.5",
        "hf-transfer==0.1.9",
        "huggingface_hub==1.33.0",
        "peft==0.21.2",
        "torch==2.14.1",
        "torchvision==0.29.1",
        "transformers==5.17.0",
        "trl==1.15.0",
        "unsloth==2026.10.7",
        "unsloth_zoo==2026.10.7",
        "wandb==0.21.0",
        "xformers==0.0.35",
    )
    .env({"HF_HOME": "/model_cache"})
    # Our own code (the shared prompt), so jobs can import `judge` on the GPU machine.
    .add_local_python_source("judge")
)
