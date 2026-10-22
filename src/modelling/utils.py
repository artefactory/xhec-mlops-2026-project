import pickle
from pathlib import Path
from typing import Any


def pickle_object(obj: Any, path: Path) -> None:
    """Save a Python object (model, vectorizer...) as a pickle file.

    Args:
        obj: The object to save.
        path: Destination of the pickle file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load a Python object from a pickle file.

    Args:
        path: Path of the pickle file.

    Returns:
        The loaded object.
    """
    with path.open("rb") as f:
        return pickle.load(f)
