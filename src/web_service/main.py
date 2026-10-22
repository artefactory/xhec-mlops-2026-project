# This module is the FastAPI app: it loads the trained model once and serves predictions.

from pathlib import Path

from fastapi import FastAPI

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import AbaloneInput, AbalonePrediction
from src.web_service.utils import load_object

MODEL_PATH = Path(__file__).parent / "local_objects" / "model.pkl"

app = FastAPI(
    title="Abalone Age Prediction",
    description="Predict the number of rings (age) of an abalone from its physical measurements.",
)

model = load_object(MODEL_PATH)


@app.get("/")
def home() -> dict:
    """Health check."""
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=AbalonePrediction, status_code=201)
def predict(payload: AbaloneInput) -> dict:
    """Predict the number of rings of one abalone."""
    return {"rings": run_inference(payload, model)}
