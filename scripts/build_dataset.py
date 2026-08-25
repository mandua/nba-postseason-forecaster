from pathlib import Path

import pandas as pd


HISTORICAL_RAW_FILE = "nba_games_raw.csv"
CURRENT_RAW_FILE = "nba_games_2025_26_raw.csv"
OUTPUT_FILE = "nba_games_clean.csv"
CURRENT_SEASON = "2025-26"

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

ID_COLUMNS = [
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
]

BOX_SCORE_COLUMNS = [
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

RAW_REQUIRED_COLUMNS = ID_COLUMNS + ["FTA"] + BOX_SCORE_COLUMNS
STANDARD_COLUMNS = ID_COLUMNS + ["SEASON", "FTA"] + BOX_SCORE_COLUMNS
BASE_GAME_COLUMNS = ["GAME_ID", "GAME_DATE", "SEASON"]


def read_raw_games(path):
    return pd.read_csv(
        path,
        dtype={
            "SEASON_ID": str,
            "TEAM_ID": str,
            "GAME_ID": str,
        },
    )


def standardize_raw_games(raw_games, source_name, season=None):
    missing_columns = [
        column
        for column in RAW_REQUIRED_COLUMNS
        if column not in raw_games.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: {missing_columns}"
        )

    games = raw_games.copy()

    if "SEASON" not in games.columns:
        if season is None:
            raise ValueError(f"{source_name} is missing SEASON.")
        games["SEASON"] = season
    elif season is not None:
        games["SEASON"] = games["SEASON"].fillna(season)

    games["GAME_ID"] = games["GAME_ID"].astype(str)
    games["TEAM_ID"] = games["TEAM_ID"].astype(str)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="raise")
    games["FTA"] = pd.to_numeric(games["FTA"], errors="raise")

    return games[STANDARD_COLUMNS].copy()


def validate_expected_seasons(games):
    seasons = set(games["SEASON"].dropna().unique())
    missing_seasons = [season for season in EXPECTED_SEASONS if season not in seasons]

    if missing_seasons:
        raise ValueError(
            "Combined dataset is missing expected seasons: "
            f"{missing_seasons}"
        )


def remove_team_game_duplicates(games):
    before_count = len(games)
    deduped = games.drop_duplicates(subset=["TEAM_ID", "GAME_ID"], keep="first")
    removed_count = before_count - len(deduped)

    return deduped.copy(), removed_count


def fill_zero_attempt_ft_pct(games):
    fixed_games = games.copy()
    missing_ft_pct = fixed_games["FT_PCT"].isna()
    zero_attempts = fixed_games["FTA"].eq(0)
    positive_attempts = fixed_games["FTA"].gt(0)
    fill_mask = missing_ft_pct & zero_attempts
    positive_attempt_missing = missing_ft_pct & positive_attempts
    unexplained_missing = missing_ft_pct & ~zero_attempts & ~positive_attempts

    if positive_attempt_missing.any():
        raise ValueError(
            "FT_PCT is missing for rows with FTA > 0. Problem rows:\n"
            f"{fixed_games.loc[positive_attempt_missing, ['GAME_ID', 'TEAM_ID', 'MATCHUP', 'FTA', 'FT_PCT']].head(20).to_string(index=False)}"
        )

    if unexplained_missing.any():
        raise ValueError(
            "FT_PCT is missing for rows that are not confirmed zero-attempt "
            "free-throw cases. Problem rows:\n"
            f"{fixed_games.loc[unexplained_missing, ['GAME_ID', 'TEAM_ID', 'MATCHUP', 'FTA', 'FT_PCT']].head(20).to_string(index=False)}"
        )

    fixed_games.loc[fill_mask, "FT_PCT"] = 0.0

    return fixed_games, int(fill_mask.sum())


def validate_two_team_rows_per_game(games):
    game_id_counts = games["GAME_ID"].value_counts()
    invalid_counts = game_id_counts[game_id_counts != 2]

    if not invalid_counts.empty:
        raise ValueError(
            "Every GAME_ID must have exactly two team rows after duplicate "
            "TEAM_ID + GAME_ID rows are removed. Problem counts:\n"
            f"{invalid_counts.sort_index().head(20).to_string()}"
        )


def add_home_away_flags(games):
    flagged_games = games.copy()
    flagged_games["IS_HOME"] = flagged_games["MATCHUP"].str.contains(
        "vs.", regex=False, na=False
    )
    flagged_games["IS_AWAY"] = flagged_games["MATCHUP"].str.contains(
        "@", regex=False, na=False
    )

    invalid_matchups = flagged_games[
        flagged_games["IS_HOME"] == flagged_games["IS_AWAY"]
    ]

    if not invalid_matchups.empty:
        raise ValueError(
            "MATCHUP must identify exactly one location using 'vs.' for home "
            "or '@' for away. Problem rows:\n"
            f"{invalid_matchups[['GAME_ID', 'TEAM_ID', 'MATCHUP']].head(20).to_string(index=False)}"
        )

    return flagged_games


def get_home_away_resolution(games):
    total_counts = games.groupby("GAME_ID").size()
    home_counts = games[games["IS_HOME"]].groupby("GAME_ID").size()
    away_counts = games[games["IS_AWAY"]].groupby("GAME_ID").size()

    return pd.DataFrame(
        {
            "team_rows": total_counts,
            "home_rows": home_counts,
            "away_rows": away_counts,
        }
    ).fillna(0).astype(int)


def find_neutral_site_game_ids(resolution):
    neutral_site_games = resolution[
        (resolution["team_rows"] == 2)
        & (resolution["home_rows"] == 0)
        & (resolution["away_rows"] == 2)
    ]

    return neutral_site_games.index.sort_values().tolist()


def validate_home_away_resolution(games):
    resolution = get_home_away_resolution(games)
    neutral_site_game_ids = find_neutral_site_game_ids(resolution)
    invalid_games = resolution[
        ((resolution["home_rows"] != 1) | (resolution["away_rows"] != 1))
        & ~resolution.index.isin(neutral_site_game_ids)
    ]

    if not invalid_games.empty:
        raise ValueError(
            "Each GAME_ID must resolve to exactly one home team and one away "
            "team unless it is a neutral-site game with two '@' rows. "
            "Problem counts:\n"
            f"{invalid_games.sort_index().head(20).to_string()}"
        )

    return neutral_site_game_ids


def validate_game_metadata(games):
    metadata_counts = games.groupby("GAME_ID")[["GAME_DATE", "SEASON"]].nunique()
    invalid_metadata = metadata_counts[
        (metadata_counts["GAME_DATE"] != 1) | (metadata_counts["SEASON"] != 1)
    ]

    if not invalid_metadata.empty:
        raise ValueError(
            "Each GAME_ID must have one GAME_DATE and one SEASON. Problem "
            "counts:\n"
            f"{invalid_metadata.sort_index().head(20).to_string()}"
        )


def prepare_side(games, prefix):
    team_columns = {
        "TEAM_ABBREVIATION": f"{prefix}_TEAM",
        "TEAM_NAME": f"{prefix}_TEAM_NAME",
    }
    stat_columns = {
        column: f"{prefix}_{column}"
        for column in BOX_SCORE_COLUMNS
    }

    columns = BASE_GAME_COLUMNS + list(team_columns) + BOX_SCORE_COLUMNS

    return games[columns].rename(columns={**team_columns, **stat_columns})


def build_game_level_dataset(team_games):
    games = add_home_away_flags(team_games)
    validate_two_team_rows_per_game(games)
    validate_game_metadata(games)
    neutral_site_game_ids = validate_home_away_resolution(games)

    model_games = games[~games["GAME_ID"].isin(neutral_site_game_ids)].copy()

    home_games = prepare_side(model_games[model_games["IS_HOME"]], "HOME")
    away_games = prepare_side(model_games[model_games["IS_AWAY"]], "AWAY")

    game_level = home_games.merge(
        away_games,
        on=BASE_GAME_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    if len(game_level) != model_games["GAME_ID"].nunique():
        raise ValueError(
            "Game-level dataset row count does not match the unique GAME_ID "
            "count after resolving home and away teams."
        )

    game_level["HOME_WIN"] = (
        game_level["HOME_PTS"] > game_level["AWAY_PTS"]
    ).astype(int)

    output_columns = (
        BASE_GAME_COLUMNS
        + ["HOME_TEAM", "HOME_TEAM_NAME"]
        + [f"HOME_{column}" for column in BOX_SCORE_COLUMNS]
        + ["AWAY_TEAM", "AWAY_TEAM_NAME"]
        + [f"AWAY_{column}" for column in BOX_SCORE_COLUMNS]
        + ["HOME_WIN"]
    )

    game_level = game_level[output_columns].sort_values(
        ["GAME_DATE", "GAME_ID"]
    )
    game_level["GAME_DATE"] = game_level["GAME_DATE"].dt.strftime("%Y-%m-%d")

    return game_level.reset_index(drop=True), neutral_site_game_ids


def print_summary(
    clean_games,
    removed_duplicate_team_rows,
    neutral_site_game_ids,
    filled_zero_attempt_ft_pct_count,
):
    print("\nBuild summary")
    print("-------------")
    print(f"Final DataFrame shape: {clean_games.shape}")
    print(f"Unique GAME_ID count: {clean_games['GAME_ID'].nunique()}")
    print("\nGames by season:")
    print(clean_games["SEASON"].value_counts().sort_index().to_string())
    print(f"\nDuplicate GAME_ID count: {clean_games['GAME_ID'].duplicated().sum()}")
    print(f"Removed duplicate TEAM_ID + GAME_ID rows: {removed_duplicate_team_rows}")
    print(
        "Zero-attempt FT_PCT values filled: "
        f"{filled_zero_attempt_ft_pct_count}"
    )
    print(f"Neutral-site games excluded: {len(neutral_site_game_ids)}")
    print("Neutral-site GAME_IDs:")
    if neutral_site_game_ids:
        print("\n".join(neutral_site_game_ids))
    else:
        print("(none)")
    print("\nMissing-value counts:")
    print(clean_games.isna().sum().to_string())
    print(f"\nEarliest GAME_DATE: {clean_games['GAME_DATE'].min()}")
    print(f"Latest GAME_DATE: {clean_games['GAME_DATE'].max()}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    historical_path = data_dir / HISTORICAL_RAW_FILE
    current_path = data_dir / CURRENT_RAW_FILE
    output_path = data_dir / OUTPUT_FILE

    historical_games = standardize_raw_games(
        read_raw_games(historical_path),
        HISTORICAL_RAW_FILE,
    )
    current_games = standardize_raw_games(
        read_raw_games(current_path),
        CURRENT_RAW_FILE,
        season=CURRENT_SEASON,
    )

    combined_games = pd.concat(
        [historical_games, current_games],
        ignore_index=True,
    )
    validate_expected_seasons(combined_games)

    combined_games, removed_duplicate_team_rows = remove_team_game_duplicates(
        combined_games
    )
    combined_games, filled_zero_attempt_ft_pct_count = fill_zero_attempt_ft_pct(
        combined_games
    )
    clean_games, neutral_site_game_ids = build_game_level_dataset(combined_games)

    clean_games.to_csv(output_path, index=False)

    print(f"Saved clean dataset to:\n{output_path}")
    print_summary(
        clean_games,
        removed_duplicate_team_rows,
        neutral_site_game_ids,
        filled_zero_attempt_ft_pct_count,
    )


if __name__ == "__main__":
    main()
