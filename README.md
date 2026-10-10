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

## Deploy the model

The service exposes `GET /health`, `POST /predict`, and `GET /metrics`. The
prediction endpoint accepts one JSON weather observation with the same feature
names as the CSV and returns the predicted label, rain probability, and model
version. Prometheus metrics include request counts, errors, latency, and
prediction counts by outcome.

Build and run the service locally after `dvc repro`:

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
