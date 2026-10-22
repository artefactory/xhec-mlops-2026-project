# Abalone Age Prediction

Predict abalone age from physical measurements: explore the data, train a
linear-regression pipeline, schedule retraining with Prefect, and serve the
saved pipeline through FastAPI. Predictions are **age in years** (`Rings + 1.5`),
not ring counts.

## Team

| Name | GitHub |
| --- | --- |
| Andreea Patarlageanu | [AndreeaPatarlageanu](https://github.com/AndreeaPatarlageanu) |
| Oliver Smith | [oliversmithOM](https://github.com/oliversmithOM) |
| Rémi Laget | [Remilaget](https://github.com/Remilaget) |
| Ryan Balech | [RyanBalech](https://github.com/RyanBalech) |

## Quick start: a prediction locally

Prerequisites: Git, **Python 3.11**, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Docker Desktop (running, Linux containers) or Docker Engine is required only
for the container section. Run all commands from the repository root.

```bash
git clone https://github.com/oliversmithOM/xhec-mlops-2026-project.git
cd xhec-mlops-2026-project
uv sync --locked --python 3.11
uv run python -m scripts.download_data
uv run python -m src.modelling.main data/abalone.csv
uv run uvicorn src.web_service.main:app --host 127.0.0.1 --port 8000
```

While Part 4 is under review, switch to `4/deploy_model_api` after cloning.
After merging, the default `main` branch contains these commands.

The download uses the same public [Kaggle dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)
as Part 1 and verifies its SHA-256. It preserves existing files. If Kaggle is
unavailable, download `abalone.csv` from that page into `data/` and rerun the
verification. No dataset or generated model is committed to Git.

Training runs the Part 3 Prefect flow and writes
`src/web_service/local_objects/model.pkl`. Without `PREFECT_API_URL`, Prefect
starts a temporary local server for the run. The API itself does not need a
Prefect or MLflow server.

Open **http://127.0.0.1:8000/docs**, expand `POST /predict`, click **Try it out**,
and execute the supplied example. Or, in a second terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @examples/predict.json
```

In PowerShell, use `curl.exe` in that command, or:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body (Get-Content examples/predict.json -Raw)
```

The response contains `age_years` and `model_sha256` (the loaded artifact's
checksum). `GET /health` confirms the model is loaded; `/docs` and
`/openapi.json` document the API. Stop the local server with `Ctrl+C`.

## Run the API in Docker

Complete download and training above first. The model is mounted read-only,
not baked into the image. The build excludes datasets, models, notebooks,
Git metadata, and local environments.

```bash
docker build -f Dockerfile.app -t abalone-api .
```

PowerShell:

```powershell
docker run --rm --name abalone-api -p 127.0.0.1:8000:8001 --mount "type=bind,source=$((Get-Location).Path)/src/web_service/local_objects,target=/app/src/web_service/local_objects,readonly" abalone-api
```

macOS/Linux:

```bash
docker run --rm --name abalone-api -p 127.0.0.1:8000:8001 --mount "type=bind,source=$PWD/src/web_service/local_objects,target=/app/src/web_service/local_objects,readonly" abalone-api
```

The same `/docs`, `/health`, and `/predict` URLs work on port **8000**. The
container listens on **8001** internally and runs as a non-root user. The
host port is bound to loopback for this local demo. To intentionally expose it
on your network, use `-p 0.0.0.0:8000:8001`. No authentication is implemented;
this is a course/demo service, not an Internet deployment.

From another terminal, verify actual HTTP predictions against the local model:

```bash
uv run python -m scripts.smoke_api
docker inspect --format '{{.State.Health.Status}}' abalone-api
docker stop abalone-api
```

`bin/run_services.sh` starts only Uvicorn and forwards shutdown signals via
`exec`. Prefect remains a separate service (see below), so port 4201 is not
exposed by this API container. Locked dependencies are installed from
`pyproject.toml` / `uv.lock`; this project does not maintain duplicate
`requirements.in` or `requirements.txt` files.

## API input and model lifecycle

`POST /predict` accepts one object with exactly the eight fields shown in
`examples/predict.json`: `sex` (`M`, `F`, or `I`), `length`, `diameter`, `height`,
`whole_weight`, `shucked_weight`, `viscera_weight`, and `shell_weight`.
Use the numeric scale in the training CSV; do not convert the measurements
to different units. All numeric values must be finite and nonnegative.
Zero height is allowed because it occurs in the original dataset.

Missing fields, extra fields (including `Rings` and `Age`), negative values,
numeric strings, booleans, and unknown sex codes return HTTP **422**. These
checks validate the input format; they do not establish biological plausibility
or guarantee reliable extrapolation. Predictions are not clipped or rounded.

The entire fitted encoder/scaler/regressor is loaded **once at startup**.
Missing, corrupt, unfitted, or incompatible models fail startup with a clear
error rather than returning fabricated predictions. Set `MODEL_PATH` to use
another trusted local artifact (and mount that path when using Docker).
Never load pickle files from untrusted sources.

After retraining, restart the API to load the new file. `/health` and prediction
responses expose the loaded model hash, so you can verify which version is
being served. Scheduled retraining does not silently hot-reload a live API.

## Part 1: exploration and MLflow experiments

```bash
uv sync --locked --python 3.11 --group notebook
```

In VS Code, select the project's `.venv` interpreter as the notebook kernel.
Run `notebooks/eda.ipynb`, then `notebooks/modelling.ipynb`, top to bottom.
EDA covers distributions, missing values, duplicates, categorical counts,
outliers, and correlations. The modelling notebook compares the baseline
with a log-target experiment; both runs are logged locally.

```bash
uv run --group notebook mlflow ui --backend-store-uri sqlite:///data/mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. The SQLite database and artifacts live under ignored
`data/`. Use the same database path when launching the UI; the former README
template's `notebooks/mlflow.db` path is no longer used.

## Parts 2 and 3: training and scheduled retraining

`src/modelling/` separates preprocessing, training/evaluation, serialization,
and prediction. The pipeline one-hot encodes categoricals, scales numeric
inputs, and fits linear regression. Both target columns are excluded from
features. Preprocessing fits only on the training split.

```bash
uv run python -m src.modelling.main data/abalone.csv
uv run python -m src.modelling.predicting src/web_service/local_objects/model.pkl data/abalone.csv
```

Use `--model-path path/to/model.pkl` on the training command to change the
output. The prediction CLI accepts raw CSV measurement columns (with spaces
and capital letters as in Kaggle); the HTTP API uses snake_case field names.
The saved model is the training-split model used for evaluation, not a refit
on all rows.

For scheduled retraining, start the Prefect server in one terminal:

```bash
uv run prefect server start --host 127.0.0.1 --port 4200
```

In a second terminal, from the repository root:

```powershell
# PowerShell
$env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
uv run python -m src.modelling.deployment
```

```bash
# macOS/Linux
PREFECT_API_URL=http://127.0.0.1:4200/api uv run python -m src.modelling.deployment
```

Keep both terminals running. The deployment retrains daily at **02:00 UTC**
and pauses when its runner stops. View runs at http://127.0.0.1:4200. To trigger
one immediately, set `PREFECT_API_URL` in a third terminal as above, then run:

```bash
uv run prefect deployment run "Abalone model training/daily-abalone-retraining" --watch
```

Stop the API before retraining for a coordinated update, then start it again
after the flow finishes. Stop the server/runner with `Ctrl+C` when finished.

## Results and limitations

The 4,177-row dataset has no missing values or duplicate rows. We preserve
the Part 1 rows, including outliers and zero heights. With a reproducible
80/20 split (`random_state=42`), 836 rows are held out:

| Model | RMSE (years) | MAE (years) | R² |
| --- | ---: | ---: | ---: |
| Linear regression baseline (served) | 2.212 | 1.593 | 0.548 |
| Log-target experiment | 2.220 | 1.570 | 0.545 |

The baseline is retained for its slightly lower RMSE and simplicity. These
are single-split results, not guarantees for new populations. Linear
regression may extrapolate poorly outside the training distribution. There
is no automatic model selection, model promotion, or prediction interval.

## Checks and continuous integration

Each teammate runs `uv run pre-commit install` once in their own clone.

```bash
uv run pre-commit run --all-files
uv run python -m unittest discover -s tests
```

GitHub runs checks on pushes and pull requests: formatting/linting, training
and API tests, Docker build, and an actual container HTTP smoke test. See the
**Actions** tab. CI uses a deterministic synthetic model fixture for the
container test, so it does not depend on Kaggle availability. Real-data
accuracy is evaluated separately by the training command.

If GNU Make is installed, `make setup`, `make data`, `make train`, `make api`,
`make test`, `make check`, and `make docker-build` wrap the same commands.
Make is optional; all direct commands above work in PowerShell.

## Repository layout and contributions

- `notebooks/`: Part 1 EDA and MLflow experiments.
- `src/modelling/`: Parts 2–3 training modules, Prefect flow and deployment.
- `src/web_service/`: Part 4 validated API and model loading.
- `Dockerfile.app`, `bin/run_services.sh`: container build and startup.
- `scripts/`: verified data download and live API smoke check.
- `tests/`: training, serialization, API validation, and lifecycle tests.
- `examples/predict.json`: runnable API request.
- `pyproject.toml`, `uv.lock`: Part 0 dependency management.
- `.github/workflows/ci.yaml`: checks and container verification.

All assessed work must be merged into `main`. Every student must contribute
meaningful commits using their GitHub-linked identity. Merge pull requests
with **Create a merge commit**, never squash, to preserve individual history.
