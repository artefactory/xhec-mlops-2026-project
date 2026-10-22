import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.feature_extraction import DictVectorizer

from src.web_service.lib.models import InputData, PredictionOut

# Biological convention: age in years = number of rings + 1.5
RINGS_TO_AGE_OFFSET = 1.5


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features used at training time.

    Must stay identical to `add_features` in src/modelling/preprocessing.py,
    otherwise the DictVectorizer will not see the features the model was trained on.

    Args:
        df: The abalone measurements.

    Returns:
        A copy of df with shell_ratio, meat_ratio and volume columns.
    """
    df = df.copy()
    df["shell_ratio"] = df["shell_weight"] / df["whole_weight"]
    df["meat_ratio"] = df["shucked_weight"] / df["whole_weight"]
    df["volume"] = df["length"] * df["diameter"] * df["height"]
    return df


def run_inference(input_data: InputData, dv: DictVectorizer, model: RegressorMixin) -> PredictionOut:
    """Predict the number of rings (and age) of one abalone.

    Args:
        input_data: The validated measurements sent to the API.
        dv: The DictVectorizer fitted during training.
        model: The trained regression model.

    Returns:
        The predicted number of rings and age.
    """
    df = add_features(pd.DataFrame([input_data.model_dump()]))
    x = dv.transform(df.to_dict(orient="records"))
    rings = float(model.predict(x)[0])
    return PredictionOut(rings=round(rings, 2), age=round(rings + RINGS_TO_AGE_OFFSET, 2))
