"""Train and save the abalone age prediction pipeline."""

import argparse
import json
from pathlib import Path

import pandas as pd
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import split_features_target
from .training import evaluate_model, train_model
from .utils import pickle_object

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINSET_PATH = PROJECT_ROOT / "data" / "abalone.csv"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "web_service/local_objects/model.pkl"
)


@task(name="Load and split data")
def load_and_split_data(
    trainset_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the training CSV and create a reproducible held-out split."""
    features, target = split_features_target(pd.read_csv(trainset_path))
    return train_test_split(features, target, test_size=0.2, random_state=42)


@task(name="Train and evaluate model")
def fit_and_evaluate_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, float]]:
    """Fit the baseline pipeline and evaluate it on the held-out split."""
    model = train_model(x_train, y_train)
    return model, evaluate_model(model, x_test, y_test)


@task(name="Save trained model")
def save_model(model: Pipeline, model_path: Path) -> None:
    """Persist the fitted model for use by the prediction service."""
    pickle_object(model, model_path)


@flow(name="Abalone model training")
def train_abalone_model(
    trainset_path: Path = DEFAULT_TRAINSET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, float]:
    """Train, evaluate, and save the Abalone age prediction pipeline."""
    x_train, x_test, y_train, y_test = load_and_split_data(trainset_path)
    model, metrics = fit_and_evaluate_model(x_train, x_test, y_train, y_test)
    save_model(model, model_path)
    return metrics


def main(trainset_path: Path, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Run the training flow and report held-out metrics."""
    metrics = train_abalone_model(trainset_path, model_path)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainset_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    main(args.trainset_path, args.model_path)
