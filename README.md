<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

**Authors**

| Name | GitHub |
|------|--------|
| Augustin Hourquet | [@AugustinHourquet](https://github.com/AugustinHourquet) |
| Célia Morais | [@celiamorais](https://github.com/celiamorais) |
| Ion Goursaud de Merlis | [@imerlis](https://github.com/imerlis) |
| Lucca Vettori | [@lugavt](https://github.com/lugavt) |

## 🎯 Project Overview

This project predicts the age of an abalone (a type of sea snail) from simple physical measurements, instead of the traditional time-consuming method of counting the shell rings under a microscope.

It contains:
- **A training pipeline** ([src/modelling](src/modelling)): the data is cleaned, a Random Forest is trained and saved as a pickle. The steps are orchestrated with Prefect, which can also retrain the model every week.
- **A prediction API** ([src/web_service](src/web_service)): a FastAPI app that validates the inputs with Pydantic and returns the predicted number of rings. It runs in a Docker container.

The model predicts the number of rings; the age of the abalone in years is about rings + 1.5.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Your Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

> **Note**: the dataset is not versioned in this repository. Download it and place it in the `data/` folder as `data/abalone.csv`.


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python **3.11**

### Set up your development environment:

   ```bash
   # To install uv if not already on your machine
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # or
   brew install uv

   # Create virtual environment using uv
   uv sync --all-groups

   # Install pre-commit hooks for code quality
   uv run pre-commit install
   ```

## 🔁 Training pipeline with Prefect

The training steps are Prefect tasks run by the flow in [src/modelling/flow.py](src/modelling/flow.py). The model is saved as a pickle in `src/web_service/local_objects/model.pkl`.

All the commands below are run from the root of the repository.

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and save it as `data/abalone.csv`.

2. Start the Prefect server (keep this terminal open):
   ```bash
   uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   uv run prefect server start
   ```
   The Prefect UI is available at http://127.0.0.1:4200.

3. In another terminal, run the training flow:
   ```bash
   uv run python src/modelling/flow.py
   ```

4. To retrain the model every week (Sunday at midnight), start the deployment (keep this terminal open):
   ```bash
   uv run python src/modelling/deployment.py
   ```
   To launch a run right away:
   ```bash
   uv run prefect deployment run "train-model-flow/weekly-retraining"
   ```

The flow runs and their tasks can be seen in the Prefect UI.


## 🌐 Prediction API

The API is a FastAPI app ([src/web_service/main.py](src/web_service/main.py)) with two endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict the number of rings of one abalone |

It loads the model saved by the training pipeline in `src/web_service/local_objects/model.pkl`, so after retraining the model, restart the API (and rebuild the Docker image) to use the new one.

### Run the API with Docker

Prerequisite: [Docker](https://docs.docker.com/get-docker/) installed and running.

1. Build the image (from the root of the repository):
   ```bash
   docker build -f Dockerfile.app -t abalone-api .
   ```

2. Run the container:
   ```bash
   docker run -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
   ```
   The container starts the API (port 8001 in the container, 8000 on your machine) and a Prefect server (port 4201 in the container, 4200 on your machine). Stop any Prefect server already running on your machine first, as it also uses port 4200.

3. Open:
   - the interactive API documentation at http://localhost:8000/docs, where you can try `/predict` with the "Try it out" button
   - the Prefect UI at http://localhost:4200

### Run the API locally (without Docker)

```bash
uv run uvicorn src.web_service.main:app --reload --port 8000
```

The documentation is then available at http://localhost:8000/docs.

### Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sex": "M", "length": 0.455, "diameter": 0.365, "height": 0.095, "whole_weight": 0.514, "shucked_weight": 0.2245, "viscera_weight": 0.101, "shell_weight": 0.15}'
```

The API answers with the predicted number of rings, e.g. `{"rings": 10}`.

The inputs are validated with Pydantic ([src/web_service/lib/models.py](src/web_service/lib/models.py)): `sex` must be `M`, `F` or `I` (infant) and all the measurements must be positive numbers (as in the Kaggle dataset, lengths are in mm and weights in grams, divided by 200). An invalid request is rejected with a `422` error that explains which field is wrong.

## 📁 Project Structure

```
├── bin/run_services.sh           # starts the Prefect server and the API in the container
├── data/                         # the dataset goes here (not versioned)
├── Dockerfile.app                # Docker image of the API
├── notebooks/                    # EDA and modelling experiments (MLflow)
├── pyproject.toml / uv.lock      # dependencies
└── src/
    ├── modelling/                # training pipeline
    │   ├── preprocessing.py      # load, clean and build the features
    │   ├── training.py           # train and evaluate the Random Forest
    │   ├── predicting.py         # predict with a trained model
    │   ├── flow.py               # Prefect training flow
    │   ├── deployment.py         # weekly retraining deployment
    │   └── main.py               # training without Prefect
    └── web_service/              # prediction API
        ├── main.py               # FastAPI app
        ├── utils.py              # load the pickled model
        ├── lib/models.py         # Pydantic input and output models
        ├── lib/inference.py      # request -> features -> prediction
        └── local_objects/        # trained model (model.pkl)
```

## 💡 Development

### Managing dependencies

Dependencies are managed with uv: the training and shared dependencies are in `dependencies`, the API ones (FastAPI, uvicorn) in the `app` group and the development tools in the `dev` group of [pyproject.toml](pyproject.toml). Add a package with:

```bash
uv add <package>==<version>                # or: uv add --group app <package>
```

Then sync the environment:

```bash
uv sync --all-groups
```

### Code quality

The pre-commit hooks (installed during the setup) lint and format the code with ruff and check that `uv.lock` is up to date at every commit. They also run in the CI on every push and pull request. To run them on all the files:

```bash
uv run pre-commit run --all-files
```
