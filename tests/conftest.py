"""Keeps the test suite away from the real Modal and Hugging Face settings (AGENTS.md rule 10).

pytest loads this file before any test module, so it runs before any test imports
`modal`. Modal reads its config path once, when it is imported (`modal/config.py`:
MODAL_CONFIG_PATH, else `~/.modal.toml`), so pointing it at an empty file in a
temporary folder here means no test can read the real one. Modal and Hugging Face
settings from the shell are removed, and Hugging Face's cache points at the same
temporary folder.
"""

import atexit
import os
import shutil
import tempfile

SCRATCH = tempfile.mkdtemp(prefix="model-training-tests-")
atexit.register(shutil.rmtree, SCRATCH, ignore_errors=True)

for key in list(os.environ):
    if key.startswith(("MODAL_", "HF_", "HUGGING_FACE_", "HUGGINGFACE_")):
        del os.environ[key]

MODAL_CONFIG = os.path.join(SCRATCH, "modal.toml")
open(MODAL_CONFIG, "w").close()
os.environ["MODAL_CONFIG_PATH"] = MODAL_CONFIG
os.environ["HF_HOME"] = os.path.join(SCRATCH, "huggingface")
