# Utility functions of the web service.

import pickle
from pathlib import Path
from typing import Any


def load_object(path: Path) -> Any:
    """Load a pickled object (e.g. the trained model) from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
