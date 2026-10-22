.PHONY: install lint mlflow-ui

install:
	uv sync
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

mlflow-ui:
	uv run mlflow ui --backend-store-uri mlruns
