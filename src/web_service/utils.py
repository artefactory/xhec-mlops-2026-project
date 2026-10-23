"""Load a trusted local model artifact and identify its exact contents."""

import hashlib
import pickle
from pathlib import Path


def load_object(path: Path) -> tuple[object, str]:
    """Only load pickles created by this project: pickle can execute code."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found at {path}. Train with "
            "'uv run python -m src.modelling.main data/abalone.csv', "
            "or set MODEL_PATH to a trusted fitted pipeline."
        )
    content = path.read_bytes()
    return pickle.loads(content), hashlib.sha256(content).hexdigest()
