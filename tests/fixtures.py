"""Small offline fixture for API tests and the container smoke test."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling.training import train_model
from src.modelling.utils import pickle_object
from src.web_service.lib.inference import FEATURE_COLUMNS


def training_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    features = pd.DataFrame(
        rng.uniform(0.05, 0.9, size=(30, 7)),
        columns=list(FEATURE_COLUMNS.values())[1:],
    )
    features.insert(0, "Sex", ["M", "F", "I"] * 10)
    target = 5 + 10 * features["Shell weight"] + 2 * features["Length"]
    return features, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    features, target = training_data()
    pickle_object(train_model(features, target), args.output)
