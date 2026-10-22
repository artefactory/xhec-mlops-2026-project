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

<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to your MLOps project! In this hands-on project, you'll build a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Your Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Your Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)

The EDA notebook downloads the dataset into `data/` the first time you run its Data cell. The `data/` directory is ignored by Git; do not upload its contents to GitHub.


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Set up your development environment**:
   ```bash
   # Install the locked dependencies for Parts 1 and 2
   uv sync --locked --python 3.11 --group notebook

   # Install pre-commit hooks for code quality
   uv run pre-commit install
   ```

### Running Parts 1 and 2 locally

In VS Code, select the repository's `.venv` Python interpreter as the notebook
kernel. Run `notebooks/eda.ipynb` from top to bottom first, then
`notebooks/modelling.ipynb`. EDA downloads the dataset if it is missing.
Both notebooks use the same `data/abalone.csv` as the Part 2 training command.
No environment activation is required for the `uv run` commands.

Part 1 records the baseline and optional log-target experiment in MLflow.
From the repository root, view them with:

```bash
uv run --group notebook mlflow ui --backend-store-uri sqlite:///data/mlflow.db
```

Open `http://127.0.0.1:5000`. The tracking database and model artifacts live
under ignored `data/`; they are generated locally, not shared through Git.
The optional log-target model is kept separate from the baseline. Part 2
deliberately exports the simple baseline, rather than selecting a model
automatically or logging a second set of MLflow experiments.

For Part 2 alone, `uv sync --locked --python 3.11` is sufficient; put the CSV
at `data/abalone.csv` and use the training/prediction commands above. Every
fresh clone needs its own CSV and must train once to create `model.pkl`.
You do not need Docker, a web server, or Prefect for Parts 1 and 2.

## 📋 What You'll Build

By the end of this project, you'll have created:

### 🤖 **Automated ML Pipeline**
- Training workflows using Prefect
- Automatic model retraining on schedule
- Reproducible model and data processing

### 🌐 **Prediction API**
- REST API for real-time predictions
- Input validation with Pydantic
- Docker containerization

### 📊 **Production-Ready Code**
- Clean, well-documented code
- Automated testing and formatting
- Proper error handling

## 📝 How to Work on This Project

### The Branch-by-Branch Approach

This project is organized into numbered branches, each representing a step in building your MLOps system. Think of it like a guided tutorial where each branch teaches you something new!

**Here's how it works**:

1. **Each branch = One pull request** with specific tasks
2. **Follow the numbers** (branch_0, branch_1, etc.) in order
3. **Read the PR instructions** (PR_0.md, PR_1.md, etc.) before starting
4. **Complete all TODOs** in that branch's code
5. **Create a pull request** when done
6. **Merge and move to the next branch**

### Step-by-Step Workflow

For each numbered branch:

```bash
# Switch to the branch
git checkout branch_number_i

# Get latest changes (except for branch_0)
git pull origin main
# Note: A VIM window might open - just type ":wq" to close it

# Push your branch
git push
```

Then:
1. 📖 Read the PR_i.md file carefully
2. 💻 Complete all the TODOs in the code
3. 🔧 Test your changes
4. 📤 Open **ONE** pull request to your main branch
5. ✅ Merge the pull request
6. 🔄 Move to the next branch

> **💡 Pro Tip**: Always integrate your previous work when starting a new branch (except branch_0)!

### 🔍 Understanding Pull Requests

Pull Requests (PRs) are how you propose and review changes before merging them into your main codebase. They're essential for team collaboration!

**Important**: When creating a PR, make sure you're merging into YOUR forked repository, not the original:

❌ **Wrong** (merging to original repo):
![PR Wrong](assets/PR_wrong.png)

✅ **Correct** (merging to your fork):
![PR Right](assets/PR_right.png)

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies. Install or update packages with:

```bash
uv add <package>==<version>
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### Code Quality
- The pre-commit hooks will automatically format your code
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings

## 📊 Evaluation Criteria

Your project will be evaluated on:

### 🔍 **Code Quality**
- Clean, readable code structure
- Proper naming conventions
- Good use of docstrings and type hints

### 🎨 **Code Formatting**
- Consistent style (automated with pre-commit)
- Professional presentation

### ⚙️ **Functionality**
- Code runs without errors
- All requirements implemented correctly

### 📖 **Documentation & Reproducibility**
- Clear README with setup instructions
- Team member names and GitHub usernames
- Step-by-step instructions to run everything

### 🤝 **Collaboration**
- Effective use of Pull Requests
- Good teamwork and communication

---

## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [ ] Prefect workflows for model training
- [ ] Separate modules for training and inference
- [ ] Reproducible model and encoder generation

✅ **Automated Deployment**
- [ ] Prefect deployment for regular retraining

✅ **Production API**
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [ ] Updated README with team info
- [ ] Clear setup and run instructions
- [ ] All TODOs removed from code

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
