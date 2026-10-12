import argparse
import os

import mlflow
from mlflow.entities import LoggedModelStatus
from mlflow.tracking import MlflowClient


def promote_run(
    run_id: str,
    model_name: str,
    tracking_uri: str | None = None,
) -> str:
    resolved_tracking_uri = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    )
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"Run {run_id} is not finished (status: {run.info.status}).")
    if "f1" not in run.data.metrics:
        raise ValueError(f"Run {run_id} has no F1 metric to review.")

    candidates = [
        model
        for model in client.search_logged_models(
            experiment_ids=[run.info.experiment_id], max_results=1000
        )
        if model.source_run_id == run_id
        and model.name == "model"
        and model.status == LoggedModelStatus.READY
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one ready model logged by run {run_id}; found {len(candidates)}."
        )
    logged_model = candidates[0]

    registered_names = {
        registered_model.name
        for registered_model in client.search_registered_models(max_results=1000)
    }
    if model_name not in registered_names:
        client.create_registered_model(model_name)

    existing_version = next(
        (
            version
            for version in client.search_model_versions(f"name='{model_name}'")
            if version.run_id == run_id
        ),
        None,
    )
    if existing_version is None:
        version = client.create_model_version(
            name=model_name,
            source=logged_model.artifact_location,
            run_id=run_id,
            model_id=logged_model.model_id,
            description=f"Promoted from reviewed MLflow run {run_id}",
        )
    else:
        version = existing_version

    client.set_registered_model_alias(model_name, "champion", str(version.version))
    return str(version.version)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a reviewed MLflow run and point the champion alias to it."
    )
    parser.add_argument("run_id", help="Run ID reviewed in the MLflow UI")
    parser.add_argument(
        "--model-name", default="weather-rain-classifier", help="Registered model name"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm promotion after reviewing the run's metrics",
    )
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(args.run_id)
    print(f"Run: {args.run_id}")
    print(f"Status: {run.info.status}")
    print(f"Metrics: {run.data.metrics}")
    if not args.confirm:
        print("Not promoted. Review this run, then rerun with --confirm to promote it.")
        return

    version = promote_run(args.run_id, args.model_name, tracking_uri)
    print(
        f"Registered {args.model_name} version {version}; "
        f"alias 'champion' now points to it."
    )


if __name__ == "__main__":
    main()