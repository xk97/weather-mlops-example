# Weather MLOps Example

A small DVC workflow for predicting whether it will rain tomorrow from weather
observations. The committed sample data is synthetic and shaped like the
weatherAUS dataset; it contains no copied real-world records.

## Run locally

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
uv run --python .venv\Scripts\python.exe dvc repro
uv run --python .venv\Scripts\python.exe pytest
```

The DVC pipeline cleans the input, trains a deterministic random forest, and
writes evaluation metrics to `reports/metrics.json`. The raw sample is tracked
by Git, so reproducing the pipeline does not require a DVC remote or secrets.

GitHub Actions runs the same pipeline and tests on pushes and pull requests.

## data 
/data folder holds raw and processed data, using src.data scripts.

```
.venv\Scripts\python.exe -m src.data.make_dataset data/raw/weather.csv data/processed/weather.csv
```

## CI and release triggers

The workflow has two jobs. `pipeline` runs on branch pushes, pull requests, and
manual runs; it reproduces the DVC pipeline, runs tests, and smoke-tests CSV
prediction. The `publish-release` job runs only for tags whose names start with
`v`, and only after `pipeline` succeeds. A manual run from a branch therefore
runs CI while showing `publish-release` as skipped; that is expected.

To try CI, create and push a test branch (the GitHub repository must already be
configured as the `origin` remote):

```powershell
git switch -c ci-test
git push -u origin ci-test
```

Open the repository's **Actions** tab to inspect the run. Pushing later commits
to that branch runs CI again. Opening a pull request into the default branch also
runs CI.

To try the publishing job after CI passes, push a version tag pointing to a
commit that contains the workflow:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The tag run checks the F1 threshold, creates or updates a GitHub Release with the
model and metrics, and publishes a container image to GHCR. Use a new version
tag for each release; pushing a tag makes those artifacts visible in the
repository. This publishes the artifacts but does not deploy the image to a
running host.

## Make predictions

After `dvc repro` has created the model, run the prediction CLI with an input CSV
containing `MinTemp`, `MaxTemp`, `Rainfall`, `Humidity3pm`, and `Pressure3pm`:

```powershell
.venv\Scripts\python.exe -m src.models.predict_model data/raw/weather.csv models/weather_model.joblib reports/predictions.csv
```

The output retains the input columns and adds `RainTomorrowPrediction` (`Yes` or
`No`) and `RainTomorrowProbability` (the model's probability of rain). Missing,
empty, or non-numeric feature values are rejected.

## Use DVC

Run these commands from the project root using the local Python 3.12 environment:

```powershell
.venv\Scripts\python.exe -m dvc status
.venv\Scripts\python.exe -m dvc dag
.venv\Scripts\python.exe -m dvc repro
.venv\Scripts\python.exe -m dvc metrics show
```

`dvc repro` reruns only stages affected by changed inputs or code. The sample
input (`data/raw/weather.csv`) is tracked in Git; processed data, the model, and
metrics are DVC outputs. Without a DVC remote, those outputs are cached only on
your machine, but can be regenerated from the committed source with `dvc repro`.

To share DVC outputs between machines, configure a remote supported by DVC. For
example, with an S3 bucket, install the backend and configure its location:

```powershell
uv pip install --python .venv\Scripts\python.exe "dvc[s3]"
.venv\Scripts\python.exe -m dvc remote add -d storage s3://your-bucket/weather-mlops
.venv\Scripts\python.exe -m dvc push
```

On another machine, install the same dependencies, configure access to that
remote using environment variables or your cloud provider's credential chain,
then run `python -m dvc pull`. Never commit access keys. The remote URL is saved
in `.dvc/config`; use a non-secret URL and keep credentials outside the repo.

## Track experiments with MLflow

When DVC runs the `evaluate` stage, it also creates an MLflow run in the local
SQLite store `mlflow.db`. Each run records the model parameters, a SHA-256 hash
of the processed dataset, evaluation metrics, and the trained model. Model files
are stored in MLflow's local artifact directory; the database and artifacts are
ignored by Git.

Open the experiment UI from the project root:

```powershell
uv run --python .venv\Scripts\python.exe mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then visit `http://127.0.0.1:5000` to compare runs and inspect their metrics and
models. A new run is logged whenever DVC executes the evaluation stage. GitHub
Actions uses its own temporary database and artifacts, which disappear after the
job; set `MLFLOW_TRACKING_URI` to a shared tracking server if you want CI runs to
persist and be visible locally.

### Promote a reviewed run

Model logging does not register every run. First open a run in the MLflow UI and
review its metrics, parameters, and dataset hash. Copy its run ID, then preview
it in the promotion command:

```powershell
$runId = "paste-the-run-id-from-the-MLflow-UI"
.venv\Scripts\python.exe -m src.models.promote_model $runId
```

The command prints the run status and metrics but does not change the registry.
After deciding that this run should be promoted, repeat the command with
`--confirm`:

```powershell
.venv\Scripts\python.exe -m src.models.promote_model $runId --confirm
```

This creates a version under `weather-rain-classifier` and points its `champion`
alias to that version. Re-promoting the same run reuses its version. This is a
manual local registry action; it does not deploy the model or persist a registry
to GitHub Actions. The local registry uses `mlflow.db` and the artifact store
configured for the experiment. The command does not enforce a quality threshold:
review the run's metrics and dataset hash before confirming. The synthetic sample
is for demonstrating the workflow, not for judging real model quality.

## Deploy the model

The service exposes `GET /health`, `POST /predict`, and `GET /metrics`. The
prediction endpoint accepts one JSON weather observation with the same feature
names as the CSV and returns the predicted label, rain probability, and model
version. Prometheus metrics include request counts, errors, latency, and
prediction counts by outcome.

To serve the model registered under the `champion` alias, first promote a run as
described in the MLflow section. Then start the API in PowerShell:

```powershell
.\start_service.ps1
```

The script defaults `MLFLOW_TRACKING_URI` to `sqlite:///mlflow.db` and `MODEL_URI`
to `models:/weather-rain-classifier@champion`. Existing values in the PowerShell
environment are preserved. Press Ctrl+C to stop the service.

This serves the model version currently assigned to `champion`; changing the
alias changes which version is loaded the next time the service starts. The
SQLite registry and its model artifact directory are local to this machine. For
Docker or another host, use a shared MLflow server and artifact store, or make
the local database and artifact files accessible to that environment.

For a repeatable deployment, configure `MODEL_URI` to a specific registered
version (for example, `models:/weather-rain-classifier/1`) instead of the mutable
`champion` alias. The version number should be the one you reviewed and promoted.

The Dockerfile currently bundles `models/weather_model.joblib` from the DVC
pipeline. It does not fetch the model from the MLflow registry. Build and run
this bundled-model image locally after `dvc repro`:

```powershell
docker build --build-arg MODEL_VERSION=local -t weather-mlops:local .
docker run --rm -p 8000:8000 weather-mlops:local
```

Check service health at `http://localhost:8000/health` and scrape metrics from
`http://localhost:8000/metrics`. Send a prediction with:

```powershell
$observation = @{ MinTemp = 16.1; MaxTemp = 22.4; Rainfall = 8.2; Humidity3pm = 82; Pressure3pm = 1004.6 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType 'application/json' -Body $observation
```
run locally no docker:
```
.venv\Scripts\python.exe -m uvicorn src.service:app --host 127.0.0.1 --port 8000
```

The sample service has no authentication or TLS termination. Keep it on a
trusted network for local use; before exposing it publicly, add authentication,
HTTPS, and request rate limits at the service or behind a gateway.

Push a version tag such as `v0.1.0` to trigger the release workflow. It rebuilds
and tests the pipeline, blocks publication if F1 is below `0.70`, attaches the
model and metrics to the GitHub Release, and publishes a serving image to GitHub
Container Registry (`ghcr.io/<owner>/<repo>:v0.1.0` and `:latest`). The image
contains the model and prediction service; the registry package may need its
visibility configured before your host can pull it. This example rebuilds from
the Git-tracked sample data and does not require a DVC remote.

The workflow publishes the image but does not start it on a hosting platform.
Configure your container host to pull the tagged image, expose port `8000`, and
provide a health check at `/health`. Configure Prometheus to scrape `/metrics`
and alert on request errors, latency, and unusual prediction rates. These service
metrics do not measure model accuracy; calculate delayed precision, recall, and
F1 when ground-truth outcomes arrive. Restrict release tags and retain the source
commit and metrics with each deployed model.

## Monitor a deployment

`reports/metrics.json` contains offline holdout metrics only; it is not live
monitoring. For a deployed prediction service, collect request volume, latency,
errors, missing or out-of-range inputs, and prediction rates. When actual
outcomes become available, calculate delayed quality metrics such as precision,
recall, and F1, and compare them with the offline baseline. Alert on service
health and meaningful data or quality drift, then investigate and retrain through
the versioned DVC pipeline. Avoid logging sensitive raw inputs; retain the model,
data, and code version with each set of monitoring results.
