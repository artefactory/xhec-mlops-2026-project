.PHONY: install lint mlflow-ui train-script

install:
	uv sync
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

mlflow-ui:
	uv run mlflow ui --backend-store-uri mlruns

train-script:
	uv run python src/modelling/main.py data/abalone.csv
