"""Load a trusted local pipeline and predict age from raw measurements."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def predict(model_path: Path, features: pd.DataFrame) -> np.ndarray:
    """Return age predictions; only load pickle files from trusted sources."""
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    return model.predict(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("features_path", type=Path, help="CSV of raw measurements")
    args = parser.parse_args()
    features = pd.read_csv(args.features_path).drop(
        columns=["Rings", "Age"], errors="ignore"
    )
    ages = predict(args.model_path, features)
    print(pd.DataFrame({"Age": ages}).to_csv(index=False), end="")
