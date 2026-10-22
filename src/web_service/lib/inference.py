# This module turns a validated request into model features and predicts the number of rings.

import pandas as pd
from sklearn.base import RegressorMixin

from src.modelling.predicting import predict
from src.modelling.preprocessing import build_features
from src.web_service.lib.models import AbaloneInput

# API field names -> column names of the training data
COLUMN_NAMES = {
    "sex": "Sex",
    "length": "Length",
    "diameter": "Diameter",
    "height": "Height",
    "whole_weight": "Whole weight",
    "shucked_weight": "Shucked weight",
    "viscera_weight": "Viscera weight",
    "shell_weight": "Shell weight",
}


def run_inference(payload: AbaloneInput, model: RegressorMixin) -> int:
    """Build the model features from one abalone's measurements and return the predicted rings."""
    df = pd.DataFrame([payload.model_dump()]).rename(columns=COLUMN_NAMES)
    return int(predict(model, build_features(df))[0])
