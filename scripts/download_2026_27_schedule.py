from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import ScheduleLeagueV2


SEASON = "2026-27"
EXPECTED_TEAM_COUNT = 30
EXPECTED_COMPLETE_GAME_COUNT = 1230
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

OUTPUT_FILE = "nba_schedule_2026_27.csv"

OUTPUT_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "GAME_TIME",
    "HOME_TEAM",
    "AWAY_TEAM",
    "ARENA",
    "CITY",
    "STATE_OR_COUNTRY",
    "GAME_DATE_EST",
    "GAME_TIME_EST",
    "GAME_DATETIME_EST",
    "GAME_DATE_UTC",
    "GAME_TIME_UTC",
    "GAME_DATETIME_UTC",
    "GAME_STATUS",
    "GAME_STATUS_TEXT",
    "GAME_SUBTYPE",
    "GAME_LABEL",
    "GAME_SUB_LABEL",
    "IS_NEUTRAL",
]


def download_schedule():
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            endpoint = ScheduleLeagueV2(
                league_id="00",
                season=SEASON,
                timeout=REQUEST_TIMEOUT,
            )
            return endpoint.season_games.get_data_frame()
        except Exception as error:
            last_error = error

            if attempt < MAX_RETRIES:
                wait_seconds = 2 ** (attempt - 1)
                print(
                    f"Schedule download failed on attempt {attempt}: {error}. "
                    f"Retrying in {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Unable to download {SEASON} schedule after {MAX_RETRIES} attempts."
    ) from last_error


def filter_regular_season_games(raw_schedule):
    game_ids = raw_schedule["gameId"].astype(str)
    regular_season = raw_schedule[game_ids.str.startswith("002")].copy()

    known_team_mask = (
        regular_season["homeTeam_teamTricode"].notna()
        & regular_season["awayTeam_teamTricode"].notna()
    )
    tbd_games = regular_season[~known_team_mask].copy()
    schedule = regular_season[known_team_mask].copy()

    if not tbd_games.empty:
        tbd_game_ids = sorted(tbd_games["gameId"].astype(str).tolist())
        print(
            "Warning: NBA API returned regular-season placeholder rows without "
            f"known teams. Excluding {len(tbd_games)} TBD games: "
            f"{', '.join(tbd_game_ids)}"
        )

    return schedule


def standardize_schedule(raw_schedule):
    raw_schedule = filter_regular_season_games(raw_schedule)

    schedule = pd.DataFrame(
        {
            "GAME_ID": raw_schedule["gameId"].astype(str),
            "GAME_DATE": raw_schedule["gameDate"],
            "GAME_TIME": raw_schedule["gameStatusText"],
            "HOME_TEAM": raw_schedule["homeTeam_teamTricode"],
            "AWAY_TEAM": raw_schedule["awayTeam_teamTricode"],
            "ARENA": raw_schedule["arenaName"],
            "CITY": raw_schedule["arenaCity"],
            "STATE_OR_COUNTRY": raw_schedule["arenaState"],
            "GAME_DATE_EST": raw_schedule["gameDateEst"],
            "GAME_TIME_EST": raw_schedule["gameTimeEst"],
            "GAME_DATETIME_EST": raw_schedule["gameDateTimeEst"],
            "GAME_DATE_UTC": raw_schedule["gameDateUTC"],
            "GAME_TIME_UTC": raw_schedule["gameTimeUTC"],
            "GAME_DATETIME_UTC": raw_schedule["gameDateTimeUTC"],
            "GAME_STATUS": raw_schedule["gameStatus"],
            "GAME_STATUS_TEXT": raw_schedule["gameStatusText"],
            "GAME_SUBTYPE": raw_schedule["gameSubtype"],
            "GAME_LABEL": raw_schedule["gameLabel"],
            "GAME_SUB_LABEL": raw_schedule["gameSubLabel"],
            "IS_NEUTRAL": raw_schedule["isNeutral"],
        }
    )

    schedule["GAME_DATE"] = pd.to_datetime(
        schedule["GAME_DATE"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    schedule["GAME_DATE_EST"] = pd.to_datetime(
        schedule["GAME_DATE_EST"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    return schedule.sort_values(
        ["GAME_DATE", "GAME_DATETIME_UTC", "GAME_ID"]
    ).reset_index(drop=True)


def validate_schedule(schedule):
    missing_required_columns = sorted(set(OUTPUT_COLUMNS) - set(schedule.columns))

    if missing_required_columns:
        raise ValueError(f"Schedule is missing columns: {missing_required_columns}")

    if len(schedule) != EXPECTED_COMPLETE_GAME_COUNT:
        print(
            "Warning: expected a complete 1,230-game regular-season schedule, "
            f"but the NBA API returned {len(schedule)} games with known teams. "
            "This can happen before NBA Cup-related TBD games are finalized."
        )

    teams = set(schedule["HOME_TEAM"].dropna()) | set(schedule["AWAY_TEAM"].dropna())

    if len(teams) != EXPECTED_TEAM_COUNT:
        raise ValueError(f"Expected {EXPECTED_TEAM_COUNT} NBA teams, found {len(teams)}.")

    if schedule["GAME_ID"].notna().all() and schedule["GAME_ID"].duplicated().any():
        duplicates = sorted(
            schedule.loc[schedule["GAME_ID"].duplicated(), "GAME_ID"].unique()
        )
        raise ValueError(f"GAME_ID values must be unique. Duplicates: {duplicates}")

    if schedule[["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]].isna().any().any():
        raise ValueError("Schedule has missing date or team values.")

    if schedule["HOME_TEAM"].eq(schedule["AWAY_TEAM"]).any():
        raise ValueError("Schedule contains games with identical home and away teams.")


def print_summary(schedule):
    location_columns = ["GAME_DATE", "GAME_TIME", "ARENA", "CITY", "STATE_OR_COUNTRY"]

    print("\n2026-27 NBA schedule summary")
    print("----------------------------")
    print(f"Total scheduled games: {len(schedule)}")
    print(f"Expected complete regular-season games: {EXPECTED_COMPLETE_GAME_COUNT}")
    print(f"Unique GAME_ID count: {schedule['GAME_ID'].nunique()}")
    print(f"Number of teams: {len(set(schedule['HOME_TEAM']) | set(schedule['AWAY_TEAM']))}")
    print(f"Earliest date: {schedule['GAME_DATE'].min()}")
    print(f"Latest date: {schedule['GAME_DATE'].max()}")
    print("\nMissing values for date/time/location fields:")
    print(schedule[location_columns].isna().sum().to_string())


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / OUTPUT_FILE

    raw_schedule = download_schedule()
    schedule = standardize_schedule(raw_schedule)
    validate_schedule(schedule)
    schedule.to_csv(output_path, index=False)

    print(f"Saved schedule to:\n{output_path}")
    print_summary(schedule)


if __name__ == "__main__":
    main()
