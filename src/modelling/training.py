# This module trains the model: a linear regression.
# The features must first be transformed with preprocessing.transform_features.

import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit a linear regression on the (transformed) training data."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
