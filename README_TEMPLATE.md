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
