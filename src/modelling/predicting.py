import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor


def predict(x: csr_matrix, model: RandomForestRegressor) -> np.ndarray:
    """Predict the number of rings.

    Args:
        x: Features, already transformed by the DictVectorizer.
        model: The trained model.

    Returns:
        The predicted number of rings.
    """
    return model.predict(x)
