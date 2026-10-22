# Abalone Age Prediction

## Project overview

This project builds a reproducible machine-learning system to predict abalone age from physical measurements. It takes the work from exploratory analysis and model development through to a maintainable training pipeline and prediction service.

## Work by branch

Summarize the completed work in each branch here as it is merged. Keep this section focused on what was implemented and any important results.

### Part 0 — `0/project-setup`

- [Describe the repository setup and development checks.]

### Part 1 — `1/eda-modelling`

- [Describe the exploratory data analysis, selected approach, and model results.]

### Later parts

- [Add a subsection for each remaining branch, describing its contribution.]

## Team

| Name | GitHub username |
|------|-----------------|
| [Team member name] | [@username] |
| [Team member name] | [@username] |
| [Add or remove rows as needed] | |

## Prerequisites

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Git

## Installation

Install the latest `main` branch and its dependencies.

For a fresh checkout:

```bash
git clone <team-repository-url>
cd <repository-folder>
```

For an existing checkout:

```bash
git switch main
git pull origin main
```

Then, from the repository root, sync the locked dependencies:

```bash
uv sync --locked --group notebook
```

This creates and populates the project environment, including the notebook dependencies. In VS Code, select the project `.venv` as the notebook kernel and restart it if it was already running. Then open and run the notebook cells. You do not need to activate the environment to run project commands; prefix commands with `uv run`.

The EDA notebook downloads the Abalone dataset into `data/` when you run its Data cell. The `data/` directory is ignored by Git, so its contents are not uploaded to GitHub.

## Development checks

Each teammate should enable the pre-commit checks once in their local clone:

```bash
uv run pre-commit install
```

To run the configured checks manually against all files:

```bash
uv run pre-commit run --all-files
```

## Running the project

1. Open `notebooks/eda.ipynb` and run its cells to download and explore the dataset.
2. Open `notebooks/modelling.ipynb` and run its cells to train the model and record MLflow runs.
3. To compare the recorded runs, open a terminal at the repository root and start the MLflow UI:

   ```bash
   uv run --group notebook mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --host 127.0.0.1 --port 5000
   ```

   Keep the terminal open and visit <http://127.0.0.1:5000>. If startup fails with a port access error (for example, `WinError 10013` on Windows), try another port, such as `5001`, and visit <http://127.0.0.1:5001>:

   ```bash
   uv run --group notebook mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --host 127.0.0.1 --port 5001
   ```

   Stop the UI with `Ctrl+C` in the terminal when you are done. MLflow data is local and should not be committed.

## Continuous integration

GitHub automatically runs the configured checks on pushes and pull requests. To see whether they passed, open the repository's **Actions** tab and select the relevant workflow run.

## Results

- [Summarize the final model, evaluation metrics, and any limitations.]

## Repository structure

- [Describe the main project directories and files once the implementation is complete.]


## Part 2: train from Python modules

From the repository root, place the headered Abalone CSV at `data/abalone.csv`
(the same dataset used in Part 1), then run:

```bash
uv sync --locked --python 3.11
uv run python -m src.modelling.main data/abalone.csv
```

The CSV must contain `Rings` and the original measurement columns, including
`Sex`. The target is age in years (`Rings + 1.5`). The baseline uses an 80/20
train/test split with seed 42, one-hot encoding, numeric scaling, and linear
regression. Preprocessing is fitted only on the training split. The command
prints held-out RMSE, MAE, and R² and saves the fitted pipeline to
`src/web_service/local_objects/model.pkl`. Override the destination with
`--model-path path/to/model.pkl`. Generated pickles are ignored by Git.

Verified on the 4,177-row Kaggle dataset with this split: RMSE **2.212 years**,
MAE **1.593 years**, R² **0.548** on 836 held-out rows. The saved model is the
same training-split model used for evaluation, not a refit on the full dataset.
The optional log-target experiment gets RMSE 2.220, MAE 1.570, and R² 0.545
on that same split. We retain the baseline for its lower RMSE and simplicity;
the log-target experiment has slightly lower MAE. These are held-out results
for this split, not a guarantee of performance on new populations.

To predict from a trusted local model in a new process (prints a CSV of ages):

```bash
uv run python -m src.modelling.predicting src/web_service/local_objects/model.pkl data/abalone.csv
```

The prediction command accepts a measurement-only CSV or the training CSV;
it ignores `Rings` and `Age` if present. To call prediction from Python:

```python
from pathlib import Path
import pandas as pd
from src.modelling.predicting import predict

features = pd.read_csv("data/abalone.csv").drop(
    columns=["Rings", "Age"], errors="ignore"
)
ages = predict(Path("src/web_service/local_objects/model.pkl"), features)
```

Run checks with `uv run pre-commit run --all-files` and
`uv run python -m unittest discover -s tests`. Part 2 uses the untransformed
baseline; the notebook's optional log-target experiment and MLflow tracking
remain in the notebook.

## Part 3: schedule model retraining with Prefect

After completing Part 2, make sure `data/abalone.csv` exists. Install all
project and notebook dependencies from the repository root:

```bash
uv sync --locked --group notebook
```

The training command now runs the Prefect flow immediately, records its tasks
and metrics, and saves the model:

```bash
uv run python -m src.modelling.main data/abalone.csv
```

To register a local deployment that retrains daily at 02:00 UTC, open a
terminal at the repository root and start the Prefect server:

```bash
uv run prefect server start --host 127.0.0.1
```

Keep that terminal open. In a second terminal, point the Prefect CLI and
deployment runner at the local server, then start the deployment:

PowerShell:

```powershell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
uv run python -m src.modelling.deployment
```

macOS/Linux:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api uv run python -m src.modelling.deployment
```

Keep the deployment terminal running for scheduled retraining. Open
<http://127.0.0.1:4200> to view flow runs and task states. The schedule is
paused when the deployment process stops. To trigger a run immediately, open a
third terminal, set `PREFECT_API_URL` as above, and run:

```bash
uv run prefect deployment run "Abalone model training/daily-abalone-retraining" --watch
```

Stop the server and deployment runner with `Ctrl+C` in their respective
terminals when finished. Prefect's local server and run history are for
development and should not be committed.

<div align="center">
