"""First GPU job: proves Modal can rent an L4, load Unsloth, and see the Hugging Face secret.

Run from the repo folder:
    uv run modal run scripts/hello_gpu.py
"""

import modal

app = modal.App("hello-gpu")

# Same image as Modal's Unsloth example (https://modal.com/docs/examples/unsloth_finetune),
# so later training steps start from a setup we already know works.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "wandb==0.21.0",
    )
    .env({"HF_HOME": "/model_cache"})
)


@app.function(
    image=image,
    gpu="L4",
    timeout=300,
    secrets=[modal.Secret.from_name("huggingface")],
)
def hello_gpu():
    import os

    import unsloth  # import first, as Unsloth asks
    import torch

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
    print(f"Unsloth version: {unsloth.__version__}")
    print(f"HF token present: {'yes' if os.environ.get('HF_TOKEN') else 'no'}")


@app.local_entrypoint()
def main():
    hello_gpu.remote()
