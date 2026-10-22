# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, task
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

from src.modelling.config import DV_PATH, MODEL_PATH
from src.modelling.predicting import predict
from src.modelling.preprocessing import clean_data, extract_x_y, load_data
from src.modelling.training import evaluate_model, train_model
from src.modelling.utils import pickle_object


@task(name="load-and-clean-data")
def load_and_clean_data(trainset_path: Path) -> pd.DataFrame:
    """Load the abalone CSV and apply the training-time cleaning rules."""
    return clean_data(load_data(trainset_path))


@task(name="split-data")
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test sets."""
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    return df_train, df_test


@task(name="preprocess-data")
def preprocess_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, DictVectorizer]:
    """Fit the DictVectorizer on the train set and transform train and test features."""
    x_train, y_train, dv = extract_x_y(df_train)
    x_test, y_test, _ = extract_x_y(df_test, dv=dv)
    return x_train, y_train, x_test, y_test, dv


@task(name="train-model")
def train_model_task(x_train: csr_matrix, y_train: np.ndarray) -> RandomForestRegressor:
    """Train the random forest regressor."""
    return train_model(x_train, y_train)


@task(name="evaluate-model")
def evaluate_model_task(y_test: np.ndarray, x_test: csr_matrix, model: RandomForestRegressor) -> float:
    """Compute test RMSE."""
    return evaluate_model(y_test, predict(x_test, model))


@task(name="save-artifacts")
def save_artifacts(dv: DictVectorizer, model: RandomForestRegressor) -> None:
    """Persist the DictVectorizer and trained model."""
    pickle_object(dv, DV_PATH)
    pickle_object(model, MODEL_PATH)


@flow(name="train-model", log_prints=True)
def train_model_flow(trainset_path: Path | str) -> float:
    """End-to-end training pipeline orchestrated with Prefect."""
    path = Path(trainset_path)
    df = load_and_clean_data(path)
    df_train, df_test = split_data(df)
    x_train, y_train, x_test, y_test, dv = preprocess_data(df_train, df_test)
    model = train_model_task(x_train, y_train)
    rmse = evaluate_model_task(y_test, x_test, model)
    print(f"Test RMSE: {rmse:.3f}")
    save_artifacts(dv, model)
    return rmse


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle).

    Args:
        trainset_path: Path of the abalone CSV file.
    """
    train_model_flow(trainset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=Path, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
