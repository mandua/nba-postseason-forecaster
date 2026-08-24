from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "nba_features.csv"
FEATURE_COLUMNS_FILE = "feature_columns.joblib"
OUTPUT_MODEL_FILE = "nba_win_model.joblib"

PROTECTED_MODEL_FILES = {
    "nba_win_model_holdout.joblib",
}

EXPECTED_SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


def load_training_data(data_path, feature_columns_path):
    data = pd.read_csv(data_path, dtype={"GAME_ID": str})
    feature_columns = joblib.load(feature_columns_path)

    return data, feature_columns


def validate_training_data(data, feature_columns):
    missing_features = [
        feature for feature in feature_columns if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    if data["GAME_ID"].nunique() != len(data):
        raise ValueError("GAME_ID must be unique before production retraining.")

    if data["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate GAME_ID rows are present.")

    target_values = set(data["HOME_WIN"].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(f"HOME_WIN contains unexpected values: {target_values}")

    seasons = sorted(data["SEASON"].dropna().unique().tolist())

    if seasons != EXPECTED_SEASONS:
        raise ValueError(
            f"Expected seasons {EXPECTED_SEASONS}, but found {seasons}."
        )

    if len(data[feature_columns].columns) != len(feature_columns):
        raise ValueError(
            "Feature count does not match models/feature_columns.joblib."
        )


def build_production_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )


def validate_output_path(output_path):
    if output_path.name in PROTECTED_MODEL_FILES:
        raise ValueError(f"Refusing to overwrite protected model: {output_path}")


def print_training_summary(data, feature_columns, output_path):
    print("\nProduction retraining summary")
    print("--------------------------------")
    print(f"Total training rows: {len(data)}")
    print(f"Seasons included: {sorted(data['SEASON'].unique().tolist())}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Target home-win rate: {data['HOME_WIN'].mean():.4f}")
    print(f"Saved model path: {output_path}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / INPUT_FILE
    feature_columns_path = project_root / "models" / FEATURE_COLUMNS_FILE
    output_path = project_root / "models" / OUTPUT_MODEL_FILE

    validate_output_path(output_path)

    data, feature_columns = load_training_data(data_path, feature_columns_path)
    validate_training_data(data, feature_columns)

    x_train = data[feature_columns]
    y_train = data["HOME_WIN"]

    model = build_production_pipeline()
    model.fit(x_train, y_train)

    joblib.dump(model, output_path)
    print_training_summary(data, feature_columns, output_path)


if __name__ == "__main__":
    main()
