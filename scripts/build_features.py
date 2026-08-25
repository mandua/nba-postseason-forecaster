from pathlib import Path

import pandas as pd


INPUT_FILE = "nba_games_clean.csv"
OUTPUT_FILE = "nba_features.csv"
EXPECTED_ROW_COUNT = 9509

KEEP_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "HOME_TEAM",
    "AWAY_TEAM",
    "HOME_WIN",
]

BOX_SCORE_STATS = [
    "PTS",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
]

TEAM_GAME_COLUMNS = [
    "TEAM",
    "GAME_ID",
    "GAME_DATE",
    "SEASON",
    "IS_HOME",
    "WIN",
    "PTS",
    "PTS_ALLOWED",
    "POINT_DIFF",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
]

ROLLING_WINDOWS = [5, 10, 20]

ROLLING_FEATURE_SPECS = [
    ("WIN", "WIN_PCT"),
    ("PTS", "AVG_PTS"),
    ("PTS_ALLOWED", "AVG_PTS_ALLOWED"),
    ("POINT_DIFF", "AVG_POINT_DIFF"),
    ("FG_PCT", "AVG_FG_PCT"),
    ("FG3_PCT", "AVG_FG3_PCT"),
    ("FT_PCT", "AVG_FT_PCT"),
    ("REB", "AVG_REB"),
    ("AST", "AVG_AST"),
    ("STL", "AVG_STL"),
    ("BLK", "AVG_BLK"),
    ("TOV", "AVG_TOV"),
]

SEASON_FEATURE_SPECS = [
    ("WIN", "SEASON_WIN_PCT"),
    ("PTS", "SEASON_AVG_PTS"),
    ("PTS_ALLOWED", "SEASON_AVG_PTS_ALLOWED"),
    ("POINT_DIFF", "SEASON_AVG_POINT_DIFF"),
]


def load_games(input_path):
    games = pd.read_csv(input_path, dtype={"GAME_ID": str})
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="raise")
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    return games


def validate_input_games(games):
    required_columns = set(KEEP_COLUMNS)

    for side in ["HOME", "AWAY"]:
        required_columns.update({f"{side}_TEAM"})
        required_columns.update({f"{side}_{stat}" for stat in BOX_SCORE_STATS})

    missing_columns = sorted(required_columns - set(games.columns))

    if missing_columns:
        raise ValueError(f"Input is missing required columns: {missing_columns}")


def create_team_history(games):
    home_games = pd.DataFrame(
        {
            "TEAM": games["HOME_TEAM"],
            "GAME_ID": games["GAME_ID"],
            "GAME_DATE": games["GAME_DATE"],
            "SEASON": games["SEASON"],
            "IS_HOME": 1,
            "WIN": games["HOME_WIN"],
            "PTS": games["HOME_PTS"],
            "PTS_ALLOWED": games["AWAY_PTS"],
            "POINT_DIFF": games["HOME_PTS"] - games["AWAY_PTS"],
        }
    )

    away_games = pd.DataFrame(
        {
            "TEAM": games["AWAY_TEAM"],
            "GAME_ID": games["GAME_ID"],
            "GAME_DATE": games["GAME_DATE"],
            "SEASON": games["SEASON"],
            "IS_HOME": 0,
            "WIN": 1 - games["HOME_WIN"],
            "PTS": games["AWAY_PTS"],
            "PTS_ALLOWED": games["HOME_PTS"],
            "POINT_DIFF": games["AWAY_PTS"] - games["HOME_PTS"],
        }
    )

    for stat in BOX_SCORE_STATS:
        if stat in ["PTS"]:
            continue
        home_games[stat] = games[f"HOME_{stat}"]
        away_games[stat] = games[f"AWAY_{stat}"]

    team_history = pd.concat([home_games, away_games], ignore_index=True)
    team_history = team_history[TEAM_GAME_COLUMNS].sort_values(
        ["TEAM", "SEASON", "GAME_DATE", "GAME_ID"]
    )

    return team_history.reset_index(drop=True)


def add_rolling_features(team_history):
    team_history = team_history.copy()
    grouped = team_history.groupby(["TEAM", "SEASON"], group_keys=False)

    for window in ROLLING_WINDOWS:
        for source_column, feature_prefix in ROLLING_FEATURE_SPECS:
            feature_name = f"{feature_prefix}_LAST_{window}"
            team_history[feature_name] = grouped[source_column].transform(
                lambda values: values.shift(1).rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )

    return team_history


def add_season_to_date_features(team_history):
    team_history = team_history.copy()
    grouped = team_history.groupby(["TEAM", "SEASON"], group_keys=False)

    for source_column, feature_name in SEASON_FEATURE_SPECS:
        team_history[feature_name] = grouped[source_column].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )

    return team_history


def add_rest_features(team_history):
    team_history = team_history.copy()
    grouped = team_history.groupby(["TEAM", "SEASON"], group_keys=False)

    team_history["REST_DAYS"] = grouped["GAME_DATE"].diff().dt.days
    team_history["BACK_TO_BACK"] = team_history["REST_DAYS"].eq(1).astype(int)

    return team_history


def create_team_features(team_history):
    team_features = add_rolling_features(team_history)
    team_features = add_season_to_date_features(team_features)
    team_features = add_rest_features(team_features)

    feature_columns = [
        column
        for column in team_features.columns
        if column not in TEAM_GAME_COLUMNS
    ]

    return team_features[["GAME_ID", "TEAM"] + feature_columns], feature_columns


def merge_features(games, team_features, feature_columns):
    featured_games = games[KEEP_COLUMNS].copy()

    home_features = team_features.rename(
        columns={
            "TEAM": "HOME_TEAM",
            **{column: f"HOME_{column}" for column in feature_columns},
        }
    )
    away_features = team_features.rename(
        columns={
            "TEAM": "AWAY_TEAM",
            **{column: f"AWAY_{column}" for column in feature_columns},
        }
    )

    featured_games = featured_games.merge(
        home_features,
        on=["GAME_ID", "HOME_TEAM"],
        how="left",
        validate="one_to_one",
    )
    featured_games = featured_games.merge(
        away_features,
        on=["GAME_ID", "AWAY_TEAM"],
        how="left",
        validate="one_to_one",
    )

    featured_games = featured_games.sort_values(["GAME_DATE", "GAME_ID"])
    featured_games["GAME_DATE"] = featured_games["GAME_DATE"].dt.strftime("%Y-%m-%d")

    return featured_games.reset_index(drop=True)


def validate_feature_dataset(features):
    if len(features) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} rows, found {len(features)}."
        )

    if features["GAME_ID"].nunique() != len(features):
        raise ValueError("GAME_ID must remain unique in the feature dataset.")

    if features["GAME_ID"].duplicated().any():
        raise ValueError("Duplicate GAME_ID rows were introduced.")

    if features.duplicated().any():
        raise ValueError("Duplicate game rows were introduced.")

    valid_home_win_values = set(features["HOME_WIN"].dropna().unique())

    if not valid_home_win_values.issubset({0, 1}):
        raise ValueError(f"HOME_WIN contains unexpected values: {valid_home_win_values}")


def print_validation_summary(features):
    missing_by_column = features.isna().sum()
    final_feature_columns = [
        column
        for column in features.columns
        if column not in KEEP_COLUMNS
    ]

    print("\nFeature build summary")
    print("---------------------")
    print(f"Final DataFrame shape: {features.shape}")
    print(f"Unique GAME_ID count: {features['GAME_ID'].nunique()}")
    print("\nGames by season:")
    print(features["SEASON"].value_counts().sort_index().to_string())
    print(f"\nDuplicate GAME_ID count: {features['GAME_ID'].duplicated().sum()}")
    print(f"Total missing values: {features.isna().sum().sum()}")
    print("\nMissing values by column:")
    print(missing_by_column.to_string())
    print(f"\nNumber of rows for 2025-26: {(features['SEASON'] == '2025-26').sum()}")
    print(f"Earliest GAME_DATE: {features['GAME_DATE'].min()}")
    print(f"Latest GAME_DATE: {features['GAME_DATE'].max()}")
    print("\nFinal feature columns:")
    print(final_feature_columns)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    input_path = data_dir / INPUT_FILE
    output_path = data_dir / OUTPUT_FILE

    games = load_games(input_path)
    validate_input_games(games)

    team_history = create_team_history(games)
    team_features, feature_columns = create_team_features(team_history)
    features = merge_features(games, team_features, feature_columns)

    validate_feature_dataset(features)
    features.to_csv(output_path, index=False)

    print(f"Saved feature dataset to:\n{output_path}")
    print_validation_summary(features)


if __name__ == "__main__":
    main()
