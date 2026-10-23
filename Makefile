.PHONY: setup data train api test check docker-build

setup:
	uv sync --locked --python 3.11
	uv run pre-commit install

data:
	uv run python -m scripts.download_data

train:
	uv run python -m src.modelling.main data/abalone.csv

api:
	uv run uvicorn src.web_service.main:app --host 127.0.0.1 --port 8000

test:
	uv run python -m unittest discover -s tests

check:
	uv run pre-commit run --all-files

docker-build:
	docker build -f Dockerfile.app -t abalone-api .
