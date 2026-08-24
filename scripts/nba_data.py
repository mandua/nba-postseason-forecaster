from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# -----------------------------
# Seasons to download
# -----------------------------
start_year = 2018
end_year = 2025

seasons = [
    f"{year}-{str(year + 1)[-2:]}"
    for year in range(start_year, end_year)
]

all_games = []

print("Downloading NBA game logs...\n")

# -----------------------------
# Download each season
# -----------------------------
for season in seasons:
    print(f"Downloading {season}...")

    gamelog = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season"
    )

    df = gamelog.get_data_frames()[0]
    df["SEASON"] = season

    all_games.append(df)

    # Be polite to the NBA stats servers
    time.sleep(1)

# -----------------------------
# Combine all seasons
# -----------------------------
games = pd.concat(all_games, ignore_index=True)

# -----------------------------
# Preview
# -----------------------------
print("\nFirst 5 rows:")
print(games.head())

print("\nDataset shape:")
print(games.shape)

print("\nColumns:")
print(games.columns.tolist())

# -----------------------------
# Save CSV
# -----------------------------
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data"

data_folder.mkdir(exist_ok=True)

output_file = data_folder / "nba_games_raw.csv"

games.to_csv(output_file, index=False)

print(f"\nDataset saved to:\n{output_file}")