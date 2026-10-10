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
