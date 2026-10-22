from pathlib import Path

from fastapi import FastAPI

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import InputData, PredictionOut
from src.web_service.utils import load_object

# Objects produced by the training flow (src/modelling/main.py)
LOCAL_OBJECTS_DIRPATH = Path(__file__).resolve().parent / "local_objects"
MODEL_PATH = LOCAL_OBJECTS_DIRPATH / "model.pkl"
DV_PATH = LOCAL_OBJECTS_DIRPATH / "dict_vectorizer.pkl"

app = FastAPI(
    title="Abalone Age Prediction API",
    description="Predict the number of rings (and age) of an abalone from its physical measurements.",
)

model = load_object(MODEL_PATH)
dv = load_object(DV_PATH)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData) -> PredictionOut:
    return run_inference(payload, dv, model)
