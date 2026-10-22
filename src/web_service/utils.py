import pickle
from pathlib import Path
from typing import Any


def load_object(path: Path) -> Any:
    """Load a Python object (model, vectorizer...) from a pickle file.

    Args:
        path: Path of the pickle file.

    Returns:
        The loaded object.
    """
    with path.open("rb") as f:
        return pickle.load(f)
