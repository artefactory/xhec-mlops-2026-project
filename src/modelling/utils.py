"""Persist trained objects for use by the prediction service."""

import pickle
from pathlib import Path


def pickle_object(obj: object, path: Path) -> None:
    """Save an object, creating its output directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
