from config import DATA_DIRPATH, MODELS_DIRPATH
from prefect import serve
from workflows import train_model_workflow

if __name__ == "__main__":
    train_deployment = train_model_workflow.to_deployment(
        name="Weekly training",
        version="0.1.0",
        tags=["training"],
        cron="0 0 * * 1",  # every Monday at midnight
        parameters={
            "trainset_path": str(DATA_DIRPATH / "abalone.csv"),
            "artifacts_dirpath": str(MODELS_DIRPATH),
        },
    )

    serve(train_deployment)
