<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
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


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Python 3.10 or 3.11

### Team

| Name | GitHub |
|------|--------|
| Wei Chieh | [@weichieh-chou](https://github.com/weichieh-chou) |
| Martin d'Halloy | [@martdhal](https://github.com/martdhal) |
| Harshita Goyal | [@hxrshu](https://github.com/hxrshu) |
| Maximilien Lucille | [@MaximilienLuc](https://github.com/MaximilienLuc) |

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Set up your development environment**:
   ```bash
   # Install dependencies (creates .venv) and sync the lockfile
   uv sync --all-groups

   # Activate the virtual environment
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate

   # Install pre-commit hooks (runs ruff + formatting checks before each commit)
   uv run pre-commit install
   ```

4. **Verify the setup**:
   ```bash
   uv run pre-commit run --all-files
   ```

   Continuous integration runs the same checks on every push and pull request (see `.github/workflows/ci.yaml`).

5. **Download the dataset** and place it at `data/abalone.csv` (see [Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)).

### Train the model with Prefect

The training pipeline lives in `src/modelling/` and is orchestrated with Prefect `flow` and `task` objects.

**Run a one-off training** (from the project root, with the virtual environment active):

```bash
cd src/modelling
python main.py ../../data/abalone.csv
```

Pickled artifacts are written to `src/web_service/local_objects/` (`model.pkl` and `dict_vectorizer.pkl`).

**Run the Prefect UI and scheduled retraining**

1. Point the CLI at your local Prefect server (Windows users may prefer `127.0.0.1` instead of `0.0.0.0`):

   ```bash
   prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
   ```

2. Start the Prefect server (leave this terminal open):

   ```bash
   prefect server start --host 0.0.0.0
   ```

   Open the dashboard at [http://localhost:4200](http://localhost:4200).

3. In **another terminal**, serve the retraining deployment (see `cron` in `deployment.py`):

   ```bash
   cd src/modelling
   python deployment.py
   ```

4. In the UI, go to **Deployments** → **train-model**. Use **Quick Run** to trigger training immediately, then open the latest **Flow run** to inspect each task (load, split, preprocess, train, evaluate, save).

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

The repo also keeps `requirements.in` / `requirements-dev.in` (pip-tools) in sync with `pyproject.toml`. After changing `.in` files, recompile the lock files:

```bash
source .venv/bin/activate
pip install pip-tools
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt
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

Progress through **branch 3 (`3/use_prefect`)** — items below reflect what is done on this branch vs. still to come (branches 4+).

✅ **Automated Training Pipeline**
- [x] Prefect workflows for model training (`src/modelling/main.py`: `@flow` / `@task`)
- [x] Separate modules for training and inference (`training.py`, `predicting.py`, `preprocessing.py`)
- [x] Reproducible model and encoder generation (`model.pkl`, `dict_vectorizer.pkl` in `src/web_service/local_objects/`)

✅ **Automated Deployment**
- [x] Prefect deployment for regular retraining (`src/modelling/deployment.py`)

✅ **Production API** *(branch 4+)*
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [x] Updated README with team info
- [x] Clear setup and run instructions (env setup, training, Prefect UI)
- [x] All TODOs removed from code under `src/`

✅ **Dependencies (PR 3)**
- [x] `prefect` in `pyproject.toml` / `uv.lock`
- [x] `requirements.in` / `requirements-dev.in` updated and compiled to `.txt`

---

**Next step:** merge PR 3, then check out branch `4/deploy_model_api` for the API and Docker tasks.
