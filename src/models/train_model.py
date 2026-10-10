import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN


def train_model(input_path: Path, model_path: Path) -> None:
	data = pd.read_csv(input_path)
	features_train, _, target_train, _ = train_test_split(
		data[FEATURE_COLUMNS],
		data[TARGET_COLUMN],
		test_size=0.25,
		random_state=42,
		stratify=data[TARGET_COLUMN],
	)
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(features_train, target_train)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, model_path)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("input_path", type=Path)
	parser.add_argument("model_path", type=Path)
	args = parser.parse_args()
	train_model(args.input_path, args.model_path)


if __name__ == "__main__":
	main()
