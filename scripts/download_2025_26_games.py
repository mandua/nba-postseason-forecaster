from pathlib import Path
import time

from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog


SEASON = "2025-26"
SEASON_TYPE = "Regular Season"
PLAYER_OR_TEAM = "T"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def download_game_log():
    """Download the team-level NBA game log with simple retry handling."""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Downloading {SEASON} {SEASON_TYPE} team game logs...")
            gamelog = LeagueGameLog(
                season=SEASON,
                season_type_all_star=SEASON_TYPE,
                player_or_team_abbreviation=PLAYER_OR_TEAM,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return gamelog.get_data_frames()[0]
        except Exception as error:
            last_error = error
            print(f"Attempt {attempt} failed: {error}")

            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Failed to download game logs after {MAX_RETRIES} attempts."
    ) from last_error


def print_download_summary(games):
    """Print basic checks for the downloaded team-level game log."""
    game_id_counts = games["GAME_ID"].value_counts()
    unexpected_counts = game_id_counts[game_id_counts != 2]

    print("\nDownload summary")
    print("----------------")
    print(f"DataFrame shape: {games.shape}")
    print(f"Unique GAME_ID count: {games['GAME_ID'].nunique()}")
    print(f"Minimum GAME_DATE: {games['GAME_DATE'].min()}")
    print(f"Maximum GAME_DATE: {games['GAME_DATE'].max()}")
    print(f"Number of unique teams: {games['TEAM_ID'].nunique()}")
    print(f"All column names: {games.columns.tolist()}")

    if unexpected_counts.empty:
        print("\nValidation passed: each GAME_ID appears exactly twice.")
    else:
        print(
            "\nWARNING: Some GAME_ID values do not appear exactly twice. "
            "This is unexpected for team-level game logs."
        )
        print(unexpected_counts.sort_index())


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_file = project_root / "data" / "nba_games_2025_26_raw.csv"

    if output_file.exists():
        raise FileExistsError(
            f"{output_file} already exists. Remove or rename it before "
            "running this script so existing CSV files are not modified."
        )

    games = download_game_log()

    output_file.parent.mkdir(exist_ok=True)
    games.to_csv(output_file, index=False)

    print(f"\nRaw game logs saved to:\n{output_file}")
    print_download_summary(games)


if __name__ == "__main__":
    main()
