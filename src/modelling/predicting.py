import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression


def predict(x: csr_matrix, model: LinearRegression) -> np.ndarray:
    """Predict the number of rings."""
    return model.predict(x)
