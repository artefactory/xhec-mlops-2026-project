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

Clone the team repository and install its dependencies:

```bash
git clone <team-repository-url>
cd <repository-folder>
uv sync --group notebook
```

`uv sync --group notebook` creates and populates the project environment, including the notebook dependencies. In VS Code, select the project `.venv` as the notebook kernel. You do not need to activate it to run project commands; prefix commands with `uv run`.

The EDA notebook downloads the Abalone dataset into `data/` when you run its Data cell. The `data/` directory is ignored by Git and its contents should not be uploaded to GitHub.

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

- [Add instructions for running the analysis, training, or application.]
- [Document required data, configuration, and any expected outputs.]

## Continuous integration

GitHub automatically runs the configured checks on pushes and pull requests. To see whether they passed, open the repository's **Actions** tab and select the relevant workflow run.

## Results

- [Summarize the final model, evaluation metrics, and any limitations.]

## Repository structure

- [Describe the main project directories and files once the implementation is complete.]
