"""Fit and evaluate the notebook's linear regression baseline."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor


def train_model(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """Fit preprocessing and regression as one reusable prediction pipeline."""
    model = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LinearRegression())]
    )
    model.fit(features, target)
    return model


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Return held-out error in years and the coefficient of determination."""
    predictions = model.predict(features)
    return {
        "rmse": float(mean_squared_error(target, predictions) ** 0.5),
        "mae": float(mean_absolute_error(target, predictions)),
        "r2": float(r2_score(target, predictions)),
    }
