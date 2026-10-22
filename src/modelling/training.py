import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def train_model(x: csr_matrix, y: np.ndarray) -> LinearRegression:
    """Train a linear regression."""
    model = LinearRegression()
    model.fit(x, y)
    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the RMSE."""
    return root_mean_squared_error(y_true, y_pred)
