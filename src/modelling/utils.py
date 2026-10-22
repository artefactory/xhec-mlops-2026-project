import pickle
from pathlib import Path
from typing import Any


def pickle_object(obj: Any, path: Path) -> None:
    """Save an object (model or DictVectorizer) to a pickle file."""
    with Path(path).open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load a pickled object (model or DictVectorizer)."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
