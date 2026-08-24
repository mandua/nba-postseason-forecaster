from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "nba_features.csv"
EVALUATION_FILE = "model_evaluation.csv"
HOLDOUT_PREDICTIONS_FILE = "holdout_predictions.csv"
MODEL_FILE = "nba_win_model_holdout.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.joblib"

TRAINING_SEASONS = [
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]
HOLDOUT_SEASON = "2025-26"
EXPECTED_HOLDOUT_ROWS = 1225

NON_FEATURE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "HOME_TEAM",
    "AWAY_TEAM",
    "HOME_WIN",
]

PREDICTION_COLUMNS = {
    "Logistic Regression": "LOGISTIC_REGRESSION_HOME_WIN_PROB",
    "Random Forest": "RANDOM_FOREST_HOME_WIN_PROB",
    "HistGradientBoosting": "HIST_GRADIENT_BOOSTING_HOME_WIN_PROB",
}


def load_features(input_path):
    data = pd.read_csv(input_path, dtype={"GAME_ID": str})
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"], errors="raise")
    data = data.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    return data


def get_feature_columns(data):
    candidate_features = data.drop(columns=NON_FEATURE_COLUMNS)
    feature_columns = candidate_features.select_dtypes(include="number").columns.tolist()

    if not feature_columns:
        raise ValueError("No numeric model feature columns were found.")

    return feature_columns


def split_data(data):
    train = data[data["SEASON"].isin(TRAINING_SEASONS)].copy()
    holdout = data[data["SEASON"].eq(HOLDOUT_SEASON)].copy()

    return train, holdout


def validate_before_training(data, train, holdout):
    if data["GAME_ID"].nunique() != len(data):
        raise ValueError("GAME_ID must be unique before training.")

    if data["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate GAME_ID rows are present before training.")

    target_values = set(data["HOME_WIN"].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(f"HOME_WIN contains unexpected values: {target_values}")

    if train["SEASON"].eq(HOLDOUT_SEASON).any():
        raise ValueError(f"{HOLDOUT_SEASON} rows were found in training data.")

    if not set(holdout["SEASON"].unique()).issubset({HOLDOUT_SEASON}):
        raise ValueError("Holdout data contains seasons other than 2025-26.")

    if len(holdout) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_HOLDOUT_ROWS} holdout games, found {len(holdout)}."
        )

    train_seasons = sorted(train["SEASON"].unique().tolist())

    if train_seasons != TRAINING_SEASONS:
        raise ValueError(
            f"Training seasons mismatch. Expected {TRAINING_SEASONS}, "
            f"found {train_seasons}."
        )


def build_models():
    return {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=42,
        ),
    }


def evaluate_model(model_name, model, x_holdout, y_holdout):
    probabilities = model.predict_proba(x_holdout)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "MODEL": model_name,
        "ACCURACY": accuracy_score(y_holdout, predictions),
        "ROC_AUC": roc_auc_score(y_holdout, probabilities),
        "LOG_LOSS": log_loss(y_holdout, probabilities, labels=[0, 1]),
        "BRIER_SCORE": brier_score_loss(y_holdout, probabilities),
    }, probabilities


def fit_and_evaluate_models(train, holdout, feature_columns):
    x_train = train[feature_columns]
    y_train = train["HOME_WIN"]
    x_holdout = holdout[feature_columns]
    y_holdout = holdout["HOME_WIN"]
    models = build_models()
    fitted_models = {}
    holdout_probabilities = {}
    results = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[model_name] = model

        result, probabilities = evaluate_model(
            model_name,
            model,
            x_holdout,
            y_holdout,
        )
        results.append(result)
        holdout_probabilities[model_name] = probabilities

    results_table = pd.DataFrame(results).sort_values(
        ["LOG_LOSS", "BRIER_SCORE"],
        ascending=[True, True],
    )

    return fitted_models, holdout_probabilities, results_table.reset_index(drop=True)


def select_best_model(results_table):
    candidate_results = results_table[
        results_table["MODEL"].isin(PREDICTION_COLUMNS.keys())
    ].copy()

    if candidate_results.empty:
        raise ValueError("No non-dummy model results are available for selection.")

    best_row = candidate_results.sort_values(
        ["LOG_LOSS", "BRIER_SCORE", "ROC_AUC"],
        ascending=[True, True, False],
    ).iloc[0]

    explanation = (
        f"{best_row['MODEL']} was selected because it had the lowest Log Loss "
        "among non-dummy models. Brier Score was used as the tie-breaker, "
        "with ROC AUC as a secondary check."
    )

    return best_row["MODEL"], explanation


def create_holdout_predictions(holdout, holdout_probabilities):
    predictions = holdout[
        ["GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_WIN"]
    ].copy()
    predictions["GAME_DATE"] = predictions["GAME_DATE"].dt.strftime("%Y-%m-%d")

    for model_name, column_name in PREDICTION_COLUMNS.items():
        predictions[column_name] = holdout_probabilities[model_name]

    return predictions


def print_training_summary(train, holdout, feature_columns, results_table, explanation):
    print("\nTraining summary")
    print("----------------")
    print(f"Training row count: {len(train)}")
    print(f"Holdout row count: {len(holdout)}")
    print(f"Number of feature columns: {len(feature_columns)}")
    print(f"Training seasons: {sorted(train['SEASON'].unique().tolist())}")
    print(f"Holdout season: {HOLDOUT_SEASON}")
    print(f"Holdout home-win rate: {holdout['HOME_WIN'].mean():.4f}")

    print("\nModel comparison sorted by LOG_LOSS, then BRIER_SCORE:")
    print(results_table.to_string(index=False))

    print("\nSelected model")
    print("--------------")
    print(explanation)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    input_path = data_dir / INPUT_FILE
    evaluation_path = data_dir / EVALUATION_FILE
    predictions_path = data_dir / HOLDOUT_PREDICTIONS_FILE
    model_path = models_dir / MODEL_FILE
    feature_columns_path = models_dir / FEATURE_COLUMNS_FILE

    data = load_features(input_path)
    feature_columns = get_feature_columns(data)
    train, holdout = split_data(data)
    validate_before_training(data, train, holdout)

    fitted_models, holdout_probabilities, results_table = fit_and_evaluate_models(
        train,
        holdout,
        feature_columns,
    )
    best_model_name, explanation = select_best_model(results_table)
    holdout_predictions = create_holdout_predictions(holdout, holdout_probabilities)

    results_table.to_csv(evaluation_path, index=False)
    holdout_predictions.to_csv(predictions_path, index=False)
    joblib.dump(fitted_models[best_model_name], model_path)
    joblib.dump(feature_columns, feature_columns_path)

    print(f"Saved model comparison table to:\n{evaluation_path}")
    print(f"Saved holdout predictions to:\n{predictions_path}")
    print(f"Saved selected model to:\n{model_path}")
    print(f"Saved feature columns to:\n{feature_columns_path}")
    print_training_summary(train, holdout, feature_columns, results_table, explanation)


if __name__ == "__main__":
    main()
