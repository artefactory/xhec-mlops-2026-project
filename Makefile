.PHONY: install lint mlflow-ui prefect-ui train deploy

export PREFECT_API_URL = http://127.0.0.1:4200/api

install:
	uv sync
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

mlflow-ui:
	uv run mlflow ui --backend-store-uri mlruns

prefect-ui:
	uv run prefect server start --host 127.0.0.1 --port 4200

train:
	uv run python src/modelling/main.py data/abalone.csv

deploy:
	uv run python src/modelling/deployment.py
