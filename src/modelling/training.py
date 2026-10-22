import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def train_model(x_train: csr_matrix, y_train: np.ndarray) -> RandomForestRegressor:
    """Train the random forest selected in the modelling notebook.

    Args:
        x_train: Training features.
        y_train: Training target (number of rings).

    Returns:
        The trained model.
    """
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the RMSE between true and predicted number of rings.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        The root mean squared error.
    """
    return root_mean_squared_error(y_true, y_pred)
