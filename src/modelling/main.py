"""Train and save the abalone age prediction pipeline."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import split_features_target
from .training import evaluate_model, train_model
from .utils import pickle_object

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "web_service/local_objects/model.pkl"
)


def main(trainset_path: Path, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Evaluate on a held-out split and save the fitted baseline pipeline."""
    features, target = split_features_target(pd.read_csv(trainset_path))
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = train_model(x_train, y_train)
    print(json.dumps(evaluate_model(model, x_test, y_test), indent=2))
    pickle_object(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainset_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    main(args.trainset_path, args.model_path)
