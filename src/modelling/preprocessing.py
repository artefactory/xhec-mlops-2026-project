import numpy as np
import pandas as pd
from config import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET
from prefect import task
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer


@task(name="load_data")
def load_data(path: str) -> pd.DataFrame:
    """Load the abalone dataset from a csv file."""
    return pd.read_csv(path)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns to snake_case (e.g. 'Whole weight' -> 'whole_weight')."""
    return df.rename(columns=lambda col: col.lower().replace(" ", "_"))


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the abalones with a height of 0 (data errors)."""
    return df[df["height"] > 0]


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: list[str] | None = None) -> pd.DataFrame:
    """Cast categorical columns to strings so the DictVectorizer one-hot encodes them."""
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    df = df.copy()
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df


@task(name="preprocess_data")
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns, remove the outliers and encode the categorical columns."""
    df = rename_columns(df)
    df = filter_outliers(df)
    return encode_categorical_cols(df)


@task(name="extract_x_y")
def extract_x_y(
    df: pd.DataFrame, dv: DictVectorizer | None = None, with_target: bool = True
) -> tuple[csr_matrix, np.ndarray | None, DictVectorizer]:
    """Turn the dataframe into a feature matrix (and target), fitting the DictVectorizer if none is given."""
    dicts = df[CATEGORICAL_COLS + NUMERICAL_COLS].to_dict(orient="records")
    if dv is None:
        dv = DictVectorizer()
        dv.fit(dicts)
    x = dv.transform(dicts)
    y = df[TARGET].values if with_target else None
    return x, y, dv
