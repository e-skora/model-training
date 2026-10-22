"""First GPU job: proves Modal can rent an L4, load Unsloth, and see the Hugging Face secret.

Run from the repo folder:
    uv run modal run scripts/hello_gpu.py
"""

import modal

from judge.modal_image import image  # the one shared image, with its pins

app = modal.App("hello-gpu")


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
