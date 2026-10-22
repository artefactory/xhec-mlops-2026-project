"""Map API inputs to the exact feature names used by the fitted pipeline."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .models import AbaloneFeatures

FEATURE_COLUMNS = {
    "sex": "Sex",
    "length": "Length",
    "diameter": "Diameter",
    "height": "Height",
    "whole_weight": "Whole weight",
    "shucked_weight": "Shucked weight",
    "viscera_weight": "Viscera weight",
    "shell_weight": "Shell weight",
}


def validate_model(model: object) -> None:
    """Reject unfitted or incompatible artifacts before accepting traffic."""
    if not isinstance(model, Pipeline):
        raise TypeError("Model must be the fitted sklearn Pipeline from training.")
    check_is_fitted(model)
    if list(getattr(model, "feature_names_in_", [])) != list(FEATURE_COLUMNS.values()):
        raise ValueError("Model feature names do not match the eight Abalone inputs.")


def predict_age(model: Pipeline, features: AbaloneFeatures) -> float:
    """Apply the saved encoder, scaler, and regressor without refitting them."""
    frame = pd.DataFrame([features.model_dump()]).rename(columns=FEATURE_COLUMNS)
    values = np.asarray(model.predict(frame), dtype=float)
    if values.shape != (1,) or not np.isfinite(values).all():
        raise ValueError("Model did not return one finite age prediction.")
    return float(values[0])
