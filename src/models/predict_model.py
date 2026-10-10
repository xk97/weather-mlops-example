import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS


PREDICTION_COLUMN = "RainTomorrowPrediction"
PROBABILITY_COLUMN = "RainTomorrowProbability"


def predict_file(input_path: Path, model_path: Path, output_path: Path) -> pd.DataFrame:
	data = pd.read_csv(input_path)
	if data.empty:
		raise ValueError("Input CSV must contain at least one observation.")

	missing_columns = [column for column in FEATURE_COLUMNS if column not in data.columns]
	if missing_columns:
		raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

	features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
	invalid_columns = features.columns[features.isna().any()].tolist()
	if invalid_columns:
		raise ValueError(f"Missing or non-numeric values in: {', '.join(invalid_columns)}")

	model = joblib.load(model_path)
	positive_class_indices = [
		index for index, label in enumerate(model.classes_) if label == 1
	]
	if not positive_class_indices:
		raise ValueError("Loaded model does not contain the positive rain class (1).")

	predictions = model.predict(features)
	rain_probabilities = model.predict_proba(features)[:, positive_class_indices[0]]

	results = data.copy()
	results[PREDICTION_COLUMN] = ["Yes" if prediction == 1 else "No" for prediction in predictions]
	results[PROBABILITY_COLUMN] = rain_probabilities

	output_path.parent.mkdir(parents=True, exist_ok=True)
	results.to_csv(output_path, index=False)
	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict whether rain is expected tomorrow.")
	parser.add_argument("input_path", type=Path, help="CSV containing the weather features")
	parser.add_argument("model_path", type=Path, help="Trained joblib model")
	parser.add_argument("output_path", type=Path, help="Path for the prediction CSV")
	args = parser.parse_args()

	results = predict_file(args.input_path, args.model_path, args.output_path)
	print(f"Wrote {len(results)} predictions to {args.output_path}")


if __name__ == "__main__":
	main()
