import argparse
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = ["MinTemp", "MaxTemp", "Rainfall", "Humidity3pm", "Pressure3pm"]
TARGET_COLUMN = "RainTomorrow"


def build_features(input_path: Path, output_path: Path) -> None:
	data = pd.read_csv(input_path)
	data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
	data[TARGET_COLUMN] = data[TARGET_COLUMN].map({"No": 0, "Yes": 1})
	data = data.dropna(subset=[TARGET_COLUMN])
	data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	data.to_csv(output_path, index=False)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("input_path", type=Path)
	parser.add_argument("output_path", type=Path)
	args = parser.parse_args()
	build_features(args.input_path, args.output_path)


if __name__ == "__main__":
	main()
