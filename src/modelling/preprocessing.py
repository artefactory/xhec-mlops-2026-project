from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer

from src.modelling.config import TARGET_COL


def load_data(path: Path) -> pd.DataFrame:
    """Load the abalone dataset and rename its columns to snake_case.

    Args:
        path: Path of the CSV file.

    Returns:
        The dataset, e.g. "Whole weight" becomes "whole_weight".
    """
    df = pd.read_csv(path)
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the anomalous rows found during the EDA (only used for training).

    Args:
        df: The abalone data.

    Returns:
        The data without null/extreme heights and without component weights above the whole weight.
    """
    valid_rows = (
        (df["height"] > 0)
        & (df["height"] < 0.5)
        & (df["shucked_weight"] <= df["whole_weight"])
        & (df["shell_weight"] <= df["whole_weight"])
    )
    return df[valid_rows].reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features from the modelling notebook.

    Args:
        df: The abalone data.

    Returns:
        A copy of df with shell_ratio, meat_ratio and volume columns.
    """
    df = df.copy()
    df["shell_ratio"] = df["shell_weight"] / df["whole_weight"]
    df["meat_ratio"] = df["shucked_weight"] / df["whole_weight"]
    df["volume"] = df["length"] * df["diameter"] * df["height"]
    return df


def extract_x_y(
    df: pd.DataFrame, dv: DictVectorizer | None = None, with_target: bool = True
) -> tuple[csr_matrix, np.ndarray | None, DictVectorizer]:
    """Turn the dataframe into model features (and target).

    The DictVectorizer one-hot encodes the "sex" column and keeps the numerical columns as they are.

    Args:
        df: The abalone data.
        dv: A fitted DictVectorizer. If None, a new one is fitted on df (training case).
        with_target: Whether df contains the target column.

    Returns:
        The features, the target (None if with_target is False) and the DictVectorizer.
    """
    features = add_features(df.drop(columns=[TARGET_COL], errors="ignore"))
    dicts = features.to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer()
        x = dv.fit_transform(dicts)
    else:
        x = dv.transform(dicts)

    y = df[TARGET_COL].values if with_target else None
    return x, y, dv
