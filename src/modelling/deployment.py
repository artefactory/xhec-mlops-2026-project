"""Serve a daily Prefect deployment for model retraining."""

from prefect.schedules import Cron

from .main import DEFAULT_MODEL_PATH, DEFAULT_TRAINSET_PATH, train_abalone_model


def serve_daily_training() -> None:
    """Run a local runner that retrains the model daily at 02:00 UTC."""
    train_abalone_model.serve(
        name="daily-abalone-retraining",
        schedule=Cron("0 2 * * *", timezone="UTC"),
        parameters={
            "trainset_path": str(DEFAULT_TRAINSET_PATH),
            "model_path": str(DEFAULT_MODEL_PATH),
        },
        tags=["abalone", "retraining"],
        pause_on_shutdown=True,
    )


if __name__ == "__main__":
    serve_daily_training()
