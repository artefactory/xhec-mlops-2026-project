import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.feature_extraction import DictVectorizer

from src.modelling.preprocessing import add_features
from src.web_service.lib.models import InputData, PredictionOut

# Biological convention: age in years = number of rings + 1.5
RINGS_TO_AGE_OFFSET = 1.5


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
