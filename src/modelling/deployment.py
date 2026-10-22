"""Serve a Prefect deployment that retrains the abalone model on a schedule."""

from src.modelling.config import TRAINSET_PATH
from src.modelling.main import train_model_flow

if __name__ == "__main__":
    # PR 3 requires a scheduled deployment, not a specific cadence — adjust cron as you prefer.
    train_model_flow.serve(
        name="train-model",
        cron="0 0 * * 0",
        parameters={"trainset_path": str(TRAINSET_PATH)},
        tags=["training", "abalone"],
        description="Retrain the abalone age prediction model on a schedule.",
    )
