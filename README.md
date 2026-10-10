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

The GitHub Actions workflow runs CI on pushes and pull requests. To publish a
model, push a version tag such as `v0.1.0`. The release job rebuilds and tests the
pipeline, blocks publication if F1 is below `0.70`, then attaches
`weather_model.joblib` and `metrics.json` to the GitHub Release. This example
rebuilds from the Git-tracked sample data, so it does not need a DVC remote.

This publishes a versioned model artifact, not a live prediction service. To
serve predictions, package the model and prediction CLI in a service/container
and deploy it to your chosen hosting platform. Restrict release creation to
protected tags and retain the source commit and metrics with the deployed model.

## Monitor a deployment

`reports/metrics.json` contains offline holdout metrics only; it is not live
monitoring. For a deployed prediction service, collect request volume, latency,
errors, missing or out-of-range inputs, and prediction rates. When actual
outcomes become available, calculate delayed quality metrics such as precision,
recall, and F1, and compare them with the offline baseline. Alert on service
health and meaningful data or quality drift, then investigate and retrain through
the versioned DVC pipeline. Avoid logging sensitive raw inputs; retain the model,
data, and code version with each set of monitoring results.
