# This module is the training flow with Prefect: each step is a task and the flow runs them in order.

import argparse
from pathlib import Path

import pandas as pd
from prefect import flow, task
from preprocessing import TARGET, build_features, clean_data, load_data
from sklearn.ensemble import RandomForestRegressor
from training import train_model
from utils import pickle_object

TRAINSET_PATH = "data/abalone.csv"
MODEL_PATH = "src/web_service/local_objects/model.pkl"


@task
def load_data_task(path: str) -> pd.DataFrame:
    """Read the dataset."""
    return load_data(path)


@task
def clean_data_task(df: pd.DataFrame) -> pd.DataFrame:
    """Remove impossible values and outliers."""
    return clean_data(df)


@task
def build_features_task(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Get the features and the target."""
    return build_features(df), df[TARGET]


@task
def train_model_task(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """Train the model."""
    return train_model(X, y)


@task
def save_model_task(model: RandomForestRegressor, path: str) -> None:
    """Save the model as a pickle."""
    pickle_object(model, Path(path))


@flow
def train_model_flow(
    trainset_path: str = TRAINSET_PATH, model_path: str = MODEL_PATH
) -> None:
    """Train a model on the data at the given path and save it (pickle)."""
    df = load_data_task(trainset_path)
    df = clean_data_task(df)
    X, y = build_features_task(df)
    model = train_model_task(X, y)
    save_model_task(model, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument(
        "trainset_path",
        type=str,
        nargs="?",
        default=TRAINSET_PATH,
        help="Path to the training set",
    )
    args = parser.parse_args()
    train_model_flow(args.trainset_path)
