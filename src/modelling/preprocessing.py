"""Prepare features and preprocessing without leaking target information."""

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Predict age in years (Rings + 1.5), excluding both target columns."""
    if "Rings" not in data.columns:
        raise ValueError("Training CSV must contain a Rings column.")
    return data.drop(columns=["Rings", "Age"], errors="ignore"), data["Rings"] + 1.5


def build_preprocessor() -> ColumnTransformer:
    """Encode categoricals and scale numbers; fit only on training data."""
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=["object", "category", "string"]),
            ),
            ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ]
    )
