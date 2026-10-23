"""Serve the trained Abalone pipeline through a validated prediction API."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .lib.inference import predict_age, validate_model
from .lib.models import AbaloneFeatures, Health, Prediction
from .utils import load_object

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "local_objects/model.pkl"
logger = logging.getLogger(__name__)


def create_app(model_path: Path | None = None) -> FastAPI:
    """Load once at startup; restart the service to adopt a retrained model."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = model_path or Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
        model, checksum = load_object(path)
        validate_model(model)
        app.state.model = model
        app.state.model_sha256 = checksum
        logger.info("Loaded model %s (sha256=%s)", path, checksum)
        yield
        del app.state.model

    app = FastAPI(
        title="Abalone Age Prediction",
        description="Estimate age in years from eight measurements using the trained baseline.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo invalid inputs: NaN/Infinity cannot be encoded as JSON.
        errors = [
            {key: error[key] for key in ("type", "loc", "msg")}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/", response_model=Health, include_in_schema=False)
    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        """Success means startup loaded and validated the model."""
        return Health(model_sha256=request.app.state.model_sha256)

    @app.post("/predict", response_model=Prediction)
    def predict(payload: AbaloneFeatures, request: Request) -> Prediction:
        """Return an age estimate; invalid inputs receive HTTP 422."""
        try:
            age = predict_age(request.app.state.model, payload)
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(
                status_code=500, detail="Model prediction failed"
            ) from exc
        return Prediction(age_years=age, model_sha256=request.app.state.model_sha256)

    return app


app = create_app()
