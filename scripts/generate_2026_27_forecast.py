from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RANDOM_STATE = 42
NUM_SIMULATIONS = 10_000
REGULAR_SEASON_GAMES = 82

FEATURES_FILE = "nba_features.csv"
GAMES_FILE = "nba_games_clean.csv"
MODEL_FILE = "nba_win_model.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.joblib"
EVALUATION_FILE = "model_evaluation.csv"

SEASON_PREDICTIONS_FILE = "season_2026_27_predictions.csv"
PROJECTED_STANDINGS_FILE = "2027_projected_standings.csv"
PLAYOFF_PROBABILITIES_FILE = "2027_playoff_probabilities.csv"
SUMMARY_FILE = "forecast_2026_27_summary.json"

FORECAST_LABEL = "2026-27 Preseason Forecast"
SOURCE_SEASON = "2025-26"

EASTERN_CONFERENCE_TEAMS = {
    "ATL",
    "BKN",
    "BOS",
    "CHA",
    "CHI",
    "CLE",
    "DET",
    "IND",
    "MIA",
    "MIL",
    "NYK",
    "ORL",
    "PHI",
    "TOR",
    "WAS",
}

WESTERN_CONFERENCE_TEAMS = {
    "DAL",
    "DEN",
    "GSW",
    "HOU",
    "LAC",
    "LAL",
    "MEM",
    "MIN",
    "NOP",
    "OKC",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "UTA",
}

SCHEDULE_CONTEXT_FEATURES = {
    "REST_DAYS",
    "BACK_TO_BACK",
}

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

# Mirrors scripts/build_features.py, but these are evaluated after
# the completed source season rather than shifted before a historical game.
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

MODEL = None
FEATURE_COLUMNS = None
TEAM_STATES = None
MATCHUP_PROBABILITIES = {}


def load_inputs(project_root: Path):
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    features = pd.read_csv(data_dir / FEATURES_FILE, dtype={"GAME_ID": str})
    features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"], errors="raise")
    features = features.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    games = pd.read_csv(data_dir / GAMES_FILE, dtype={"GAME_ID": str})
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="raise")
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    model = joblib.load(models_dir / MODEL_FILE)
    feature_columns = joblib.load(models_dir / FEATURE_COLUMNS_FILE)

    return features, games, model, feature_columns


def get_base_feature_names(feature_columns):
    home_features = [
        column.removeprefix("HOME_")
        for column in feature_columns
        if column.startswith("HOME_")
    ]
    away_features = [
        column.removeprefix("AWAY_")
        for column in feature_columns
        if column.startswith("AWAY_")
    ]

    if home_features != away_features:
        raise ValueError("HOME and AWAY feature columns do not align.")

    return home_features


def validate_conference_assignments(teams):
    assigned_teams = EASTERN_CONFERENCE_TEAMS | WESTERN_CONFERENCE_TEAMS

    if teams != assigned_teams:
        missing = sorted(teams - assigned_teams)
        extra = sorted(assigned_teams - teams)
        raise ValueError(
            "Conference assignments do not match the 30 teams in the data. "
            f"Missing assignments: {missing}; extra assignments: {extra}"
        )


def validate_feature_game_alignment(features, games):
    feature_ids = set(features.loc[features["SEASON"].eq(SOURCE_SEASON), "GAME_ID"])
    game_ids = set(games.loc[games["SEASON"].eq(SOURCE_SEASON), "GAME_ID"])

    if feature_ids != game_ids:
        missing_from_features = sorted(game_ids - feature_ids)
        missing_from_games = sorted(feature_ids - game_ids)
        raise ValueError(
            f"{SOURCE_SEASON} GAME_ID values do not align between "
            f"{FEATURES_FILE} and {GAMES_FILE}. Missing from features: "
            f"{missing_from_features}; missing from games: {missing_from_games}"
        )


def validate_clean_games(games):
    required_columns = {
        "GAME_ID",
        "GAME_DATE",
        "SEASON",
        "HOME_TEAM",
        "AWAY_TEAM",
        "HOME_WIN",
    }

    for side in ["HOME", "AWAY"]:
        required_columns.update({f"{side}_{stat}" for stat in BOX_SCORE_STATS})

    missing_columns = sorted(required_columns - set(games.columns))

    if missing_columns:
        raise ValueError(f"{GAMES_FILE} is missing required columns: {missing_columns}")

    invalid_home_win_values = set(games["HOME_WIN"].dropna().unique()) - {0, 1}

    if invalid_home_win_values:
        raise ValueError(f"HOME_WIN contains unexpected values: {invalid_home_win_values}")


def create_completed_team_history(games):
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
        if stat == "PTS":
            continue
        home_games[stat] = games[f"HOME_{stat}"]
        away_games[stat] = games[f"AWAY_{stat}"]

    team_history = pd.concat([home_games, away_games], ignore_index=True)
    team_history = team_history[TEAM_GAME_COLUMNS].sort_values(
        ["TEAM", "SEASON", "GAME_DATE", "GAME_ID"]
    )

    return team_history.reset_index(drop=True)


def calculate_completed_season_team_states(team_history, base_feature_names):
    source_history = team_history[team_history["SEASON"].eq(SOURCE_SEASON)].copy()

    if source_history.empty:
        raise ValueError(f"No completed team history found for {SOURCE_SEASON}.")

    teams = set(source_history["TEAM"].unique())
    validate_conference_assignments(teams)

    state_rows = []

    for team, team_games in source_history.groupby("TEAM"):
        team_games = team_games.sort_values(["GAME_DATE", "GAME_ID"])
        state_row = {"TEAM": team}

        for window in ROLLING_WINDOWS:
            window_games = team_games.tail(window)

            for source_column, feature_prefix in ROLLING_FEATURE_SPECS:
                feature_name = f"{feature_prefix}_LAST_{window}"
                state_row[feature_name] = float(window_games[source_column].mean())

        for source_column, feature_name in SEASON_FEATURE_SPECS:
            state_row[feature_name] = float(team_games[source_column].mean())

        state_rows.append(state_row)

    team_states = pd.DataFrame(state_rows).set_index("TEAM").sort_index()
    source_history["REST_DAYS"] = source_history.groupby("TEAM")["GAME_DATE"].diff().dt.days
    neutral_rest_days = float(source_history["REST_DAYS"].dropna().median())

    team_states["REST_DAYS"] = neutral_rest_days
    team_states["BACK_TO_BACK"] = 0

    missing_columns = [
        feature
        for feature in base_feature_names
        if feature not in team_states.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing preseason state feature columns: {missing_columns}")

    if len(team_states) != 30:
        raise ValueError(f"Expected exactly 30 preseason team states, found {len(team_states)}.")

    team_states = team_states[base_feature_names]
    missing_state_values = team_states.isna().sum()
    missing_state_values = missing_state_values[missing_state_values > 0]

    if not missing_state_values.empty:
        raise ValueError(
            "Preseason team states contain missing feature values:\n"
            f"{missing_state_values.to_string()}"
        )

    return team_states


def build_preseason_team_states(features, games, feature_columns):
    base_feature_names = get_base_feature_names(feature_columns)

    validate_feature_game_alignment(features, games)
    validate_clean_games(games)

    team_history = create_completed_team_history(games)
    team_states = calculate_completed_season_team_states(
        team_history,
        base_feature_names,
    )

    return team_states, base_feature_names


def initialize_prediction_context(model, feature_columns, team_states):
    global MODEL, FEATURE_COLUMNS, TEAM_STATES, MATCHUP_PROBABILITIES

    MODEL = model
    FEATURE_COLUMNS = feature_columns
    TEAM_STATES = team_states
    MATCHUP_PROBABILITIES = {}


def build_model_row(home_team, away_team):
    if TEAM_STATES is None or FEATURE_COLUMNS is None:
        raise RuntimeError("Prediction context has not been initialized.")

    if home_team not in TEAM_STATES.index:
        raise ValueError(f"Unknown home team: {home_team}")

    if away_team not in TEAM_STATES.index:
        raise ValueError(f"Unknown away team: {away_team}")

    row = {}

    for column in FEATURE_COLUMNS:
        if column.startswith("HOME_"):
            row[column] = TEAM_STATES.loc[home_team, column.removeprefix("HOME_")]
        elif column.startswith("AWAY_"):
            row[column] = TEAM_STATES.loc[away_team, column.removeprefix("AWAY_")]
        else:
            raise ValueError(f"Unexpected feature column: {column}")

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_matchup(home_team, away_team):
    """Predict a single 2026-27 preseason matchup using team-state estimates."""
    if MODEL is None:
        raise RuntimeError("Prediction context has not been initialized.")

    key = (home_team, away_team)

    if key not in MATCHUP_PROBABILITIES:
        model_row = build_model_row(home_team, away_team)
        home_probability = float(MODEL.predict_proba(model_row)[0, 1])
        MATCHUP_PROBABILITIES[key] = home_probability

    home_probability = MATCHUP_PROBABILITIES[key]
    away_probability = 1.0 - home_probability

    if not np.isclose(home_probability + away_probability, 1.0, atol=1e-9):
        raise ValueError("Home and away probabilities do not sum to 1.")

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": home_probability,
        "away_win_probability": away_probability,
    }


def precompute_matchup_probabilities(teams):
    rows = []
    keys = []

    for home_team in teams:
        for away_team in teams:
            if home_team == away_team:
                continue
            rows.append(build_model_row(home_team, away_team).iloc[0])
            keys.append((home_team, away_team))

    probabilities = MODEL.predict_proba(pd.DataFrame(rows, columns=FEATURE_COLUMNS))[:, 1]

    for key, probability in zip(keys, probabilities):
        MATCHUP_PROBABILITIES[key] = float(probability)


def get_home_win_probability(home_team, away_team):
    return predict_matchup(home_team, away_team)["home_win_probability"]


def project_regular_season_records(teams):
    projected_rows = []

    for team in teams:
        implied_probs = []

        for opponent in teams:
            if opponent == team:
                continue
            home_probability = get_home_win_probability(team, opponent)
            away_probability = 1.0 - get_home_win_probability(opponent, team)
            implied_probs.append((home_probability + away_probability) / 2.0)

        projected_win_pct = float(np.mean(implied_probs))
        projected_rows.append(
            {
                "TEAM": team,
                "MODEL_STRENGTH": projected_win_pct,
                "RAW_PROJECTED_WINS": projected_win_pct * REGULAR_SEASON_GAMES,
            }
        )

    projections = pd.DataFrame(projected_rows)
    projections["PROJECTED_WINS"] = np.floor(projections["RAW_PROJECTED_WINS"]).astype(int)
    remaining_wins = int(round(projections["RAW_PROJECTED_WINS"].sum())) - int(
        projections["PROJECTED_WINS"].sum()
    )

    if remaining_wins > 0:
        remainder_order = (
            projections.assign(
                REMAINDER=projections["RAW_PROJECTED_WINS"]
                - projections["PROJECTED_WINS"]
            )
            .sort_values(["REMAINDER", "MODEL_STRENGTH", "TEAM"], ascending=[False, False, True])
            .head(remaining_wins)
            .index
        )
        projections.loc[remainder_order, "PROJECTED_WINS"] += 1

    projections["PROJECTED_WINS"] = projections["PROJECTED_WINS"].clip(
        lower=0,
        upper=REGULAR_SEASON_GAMES,
    )
    projections["PROJECTED_LOSSES"] = REGULAR_SEASON_GAMES - projections["PROJECTED_WINS"]
    projections["PROJECTED_WIN_PCT"] = projections["PROJECTED_WINS"] / REGULAR_SEASON_GAMES

    if not projections["PROJECTED_WINS"].between(0, REGULAR_SEASON_GAMES).all():
        raise ValueError("Projected wins must be between 0 and 82.")

    if not (projections["PROJECTED_WINS"] + projections["PROJECTED_LOSSES"]).eq(
        REGULAR_SEASON_GAMES
    ).all():
        raise ValueError("Projected wins and losses must total 82 for every team.")

    return projections.sort_values(
        ["PROJECTED_WINS", "MODEL_STRENGTH", "TEAM"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def add_conferences_and_seeds(projections):
    standings = projections.copy()
    standings["CONFERENCE"] = standings["TEAM"].apply(
        lambda team: "Eastern" if team in EASTERN_CONFERENCE_TEAMS else "Western"
    )
    standings = standings.sort_values(
        ["CONFERENCE", "PROJECTED_WINS", "MODEL_STRENGTH", "TEAM"],
        ascending=[True, False, False, True],
    )
    standings["PROJECTED_SEED"] = standings.groupby("CONFERENCE").cumcount() + 1

    projected_standings = standings[standings["PROJECTED_SEED"] <= 10].copy()
    projected_standings = projected_standings.rename(columns={"PROJECTED_SEED": "SEED"})

    return standings.reset_index(drop=True), projected_standings[
        [
            "CONFERENCE",
            "SEED",
            "TEAM",
            "PROJECTED_WINS",
            "PROJECTED_LOSSES",
            "PROJECTED_WIN_PCT",
        ]
    ].reset_index(drop=True)


def simulate_single_game(home_team, away_team, rng):
    home_probability = get_home_win_probability(home_team, away_team)
    return home_team if rng.random() < home_probability else away_team


def simulate_best_of_seven(home_court_team, other_team, rng):
    home_sequence = [
        home_court_team,
        home_court_team,
        other_team,
        other_team,
        home_court_team,
        other_team,
        home_court_team,
    ]
    wins = {home_court_team: 0, other_team: 0}

    for home_team in home_sequence:
        away_team = other_team if home_team == home_court_team else home_court_team
        winner = simulate_single_game(home_team, away_team, rng)
        wins[winner] += 1

        if wins[winner] == 4:
            return winner

    raise RuntimeError("Best-of-seven series did not produce a winner.")


def simulate_seeded_series(team_a, team_b, seed_by_team, rng):
    home_court_team = team_a if seed_by_team[team_a] < seed_by_team[team_b] else team_b
    other_team = team_b if home_court_team == team_a else team_a

    return simulate_best_of_seven(home_court_team, other_team, rng)


def simulate_play_in(conference_standings, rng):
    seed_to_team = dict(
        zip(conference_standings["PROJECTED_SEED"], conference_standings["TEAM"])
    )

    winner_7_8 = simulate_single_game(seed_to_team[7], seed_to_team[8], rng)
    loser_7_8 = seed_to_team[8] if winner_7_8 == seed_to_team[7] else seed_to_team[7]

    winner_9_10 = simulate_single_game(seed_to_team[9], seed_to_team[10], rng)
    eighth_seed = simulate_single_game(loser_7_8, winner_9_10, rng)

    playoff_field = {
        1: seed_to_team[1],
        2: seed_to_team[2],
        3: seed_to_team[3],
        4: seed_to_team[4],
        5: seed_to_team[5],
        6: seed_to_team[6],
        7: winner_7_8,
        8: eighth_seed,
    }
    playoff_seed_by_team = {
        team: seed
        for seed, team in playoff_field.items()
    }

    return playoff_field, playoff_seed_by_team


def simulate_conference_playoffs(conference_standings, counts, rng):
    playoff_field, seed_by_team = simulate_play_in(conference_standings, rng)

    for team in playoff_field.values():
        counts[team]["MAKE_PLAYOFFS"] += 1

    first_round_winners = {
        "1/8": simulate_seeded_series(playoff_field[1], playoff_field[8], seed_by_team, rng),
        "4/5": simulate_seeded_series(playoff_field[4], playoff_field[5], seed_by_team, rng),
        "3/6": simulate_seeded_series(playoff_field[3], playoff_field[6], seed_by_team, rng),
        "2/7": simulate_seeded_series(playoff_field[2], playoff_field[7], seed_by_team, rng),
    }

    for team in first_round_winners.values():
        counts[team]["CONF_SEMIFINALS"] += 1

    semifinal_winners = {
        "upper": simulate_seeded_series(
            first_round_winners["1/8"],
            first_round_winners["4/5"],
            seed_by_team,
            rng,
        ),
        "lower": simulate_seeded_series(
            first_round_winners["3/6"],
            first_round_winners["2/7"],
            seed_by_team,
            rng,
        ),
    }

    for team in semifinal_winners.values():
        counts[team]["CONF_FINALS"] += 1

    conference_champion = simulate_seeded_series(
        semifinal_winners["upper"],
        semifinal_winners["lower"],
        seed_by_team,
        rng,
    )
    counts[conference_champion]["NBA_FINALS"] += 1

    return conference_champion


def choose_finals_home_court(east_champion, west_champion, standings):
    lookup = standings.set_index("TEAM")
    east_row = lookup.loc[east_champion]
    west_row = lookup.loc[west_champion]

    if east_row["PROJECTED_WINS"] > west_row["PROJECTED_WINS"]:
        return east_champion, west_champion

    if west_row["PROJECTED_WINS"] > east_row["PROJECTED_WINS"]:
        return west_champion, east_champion

    if east_row["MODEL_STRENGTH"] >= west_row["MODEL_STRENGTH"]:
        return east_champion, west_champion

    return west_champion, east_champion


def simulate_playoffs(standings):
    rng = np.random.default_rng(RANDOM_STATE)
    teams = standings["TEAM"].tolist()
    counts = {
        team: {
            "MAKE_PLAYOFFS": 0,
            "CONF_SEMIFINALS": 0,
            "CONF_FINALS": 0,
            "NBA_FINALS": 0,
            "CHAMPIONSHIP": 0,
        }
        for team in teams
    }
    standings_by_conference = {
        conference: group.sort_values("PROJECTED_SEED").reset_index(drop=True)
        for conference, group in standings.groupby("CONFERENCE")
    }

    for _ in range(NUM_SIMULATIONS):
        east_champion = simulate_conference_playoffs(
            standings_by_conference["Eastern"],
            counts,
            rng,
        )
        west_champion = simulate_conference_playoffs(
            standings_by_conference["Western"],
            counts,
            rng,
        )
        finals_home_team, finals_away_team = choose_finals_home_court(
            east_champion,
            west_champion,
            standings,
        )
        champion = simulate_best_of_seven(finals_home_team, finals_away_team, rng)
        counts[champion]["CHAMPIONSHIP"] += 1

    probability_rows = []
    standings_lookup = standings.set_index("TEAM")

    for team, team_counts in counts.items():
        row = standings_lookup.loc[team]
        probability_rows.append(
            {
                "TEAM": team,
                "CONFERENCE": row["CONFERENCE"],
                "PROJECTED_SEED": int(row["PROJECTED_SEED"]),
                "MAKE_PLAYOFFS_PROB": team_counts["MAKE_PLAYOFFS"] / NUM_SIMULATIONS,
                "CONF_SEMIFINALS_PROB": team_counts["CONF_SEMIFINALS"] / NUM_SIMULATIONS,
                "CONF_FINALS_PROB": team_counts["CONF_FINALS"] / NUM_SIMULATIONS,
                "NBA_FINALS_PROB": team_counts["NBA_FINALS"] / NUM_SIMULATIONS,
                "CHAMPIONSHIP_PROB": team_counts["CHAMPIONSHIP"] / NUM_SIMULATIONS,
            }
        )

    probabilities = pd.DataFrame(probability_rows).sort_values(
        ["CHAMPIONSHIP_PROB", "NBA_FINALS_PROB", "TEAM"],
        ascending=[False, False, True],
    )

    championship_sum = probabilities["CHAMPIONSHIP_PROB"].sum()

    if not np.isclose(championship_sum, 1.0, atol=0.01):
        raise ValueError(
            "Championship probabilities should sum approximately to 1.0; "
            f"found {championship_sum:.6f}."
        )

    return probabilities.reset_index(drop=True)


def load_model_evaluation(data_dir):
    evaluation_path = data_dir / EVALUATION_FILE

    if not evaluation_path.exists():
        return None

    evaluation = pd.read_csv(evaluation_path)
    return evaluation.to_dict(orient="records")


def create_summary(standings, playoff_probabilities, model_evaluation):
    eastern_standings = standings[standings["CONFERENCE"].eq("Eastern")]
    western_standings = standings[standings["CONFERENCE"].eq("Western")]
    eastern_probs = playoff_probabilities[playoff_probabilities["CONFERENCE"].eq("Eastern")]
    western_probs = playoff_probabilities[playoff_probabilities["CONFERENCE"].eq("Western")]
    top_record = standings.sort_values(
        ["PROJECTED_WINS", "MODEL_STRENGTH", "TEAM"],
        ascending=[False, False, True],
    ).iloc[0]
    most_likely_champion = playoff_probabilities.sort_values(
        ["CHAMPIONSHIP_PROB", "NBA_FINALS_PROB", "TEAM"],
        ascending=[False, False, True],
    ).iloc[0]
    eastern_champion = eastern_probs.sort_values(
        ["NBA_FINALS_PROB", "CHAMPIONSHIP_PROB", "TEAM"],
        ascending=[False, False, True],
    ).iloc[0]
    western_champion = western_probs.sort_values(
        ["NBA_FINALS_PROB", "CHAMPIONSHIP_PROB", "TEAM"],
        ascending=[False, False, True],
    ).iloc[0]

    summary = {
        "forecast_label": FORECAST_LABEL,
        "projected_eastern_conference_1_seed": eastern_standings.sort_values(
            "PROJECTED_SEED"
        ).iloc[0]["TEAM"],
        "projected_western_conference_1_seed": western_standings.sort_values(
            "PROJECTED_SEED"
        ).iloc[0]["TEAM"],
        "highest_projected_regular_season_win_total": {
            "team": top_record["TEAM"],
            "projected_wins": int(top_record["PROJECTED_WINS"]),
        },
        "most_likely_eastern_conference_champion": eastern_champion["TEAM"],
        "most_likely_western_conference_champion": western_champion["TEAM"],
        "most_likely_nba_champion": most_likely_champion["TEAM"],
        "championship_probability": float(most_likely_champion["CHAMPIONSHIP_PROB"]),
        "model_evaluation_metrics": model_evaluation,
        "methodology_note": (
            "This 2026-27 preseason forecast uses the production forecast model, "
            "preseason team-state estimates derived from historical NBA data "
            "through the 2025-26 season, and Monte Carlo playoff simulation."
        ),
        "limitation_note": (
            "This initial preseason model does not directly account for injuries "
            "or every offseason roster change."
        ),
        "projection_note": (
            "Projected regular-season records are model-implied preseason "
            "strength projections rather than simulations of every official "
            "2026-27 scheduled game. The official schedule is used separately "
            "for scheduled-game lookup and prediction."
        ),
    }

    return summary


def write_outputs(
    data_dir,
    season_predictions,
    projected_standings,
    playoff_probabilities,
    summary,
):
    paths = {
        "season_predictions": data_dir / SEASON_PREDICTIONS_FILE,
        "projected_standings": data_dir / PROJECTED_STANDINGS_FILE,
        "playoff_probabilities": data_dir / PLAYOFF_PROBABILITIES_FILE,
        "summary": data_dir / SUMMARY_FILE,
    }

    season_predictions[
        ["TEAM", "PROJECTED_WINS", "PROJECTED_LOSSES", "PROJECTED_WIN_PCT"]
    ].to_csv(paths["season_predictions"], index=False)
    projected_standings.to_csv(paths["projected_standings"], index=False)
    playoff_probabilities.to_csv(paths["playoff_probabilities"], index=False)

    with paths["summary"].open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=4)
        summary_file.write("\n")

    return paths


def print_forecast_summary(
    team_count,
    season_predictions,
    projected_standings,
    playoff_probabilities,
    output_paths,
):
    eastern_seeds = projected_standings[projected_standings["CONFERENCE"].eq("Eastern")]
    western_seeds = projected_standings[projected_standings["CONFERENCE"].eq("Western")]
    top_championships = playoff_probabilities.sort_values(
        ["CHAMPIONSHIP_PROB", "NBA_FINALS_PROB", "TEAM"],
        ascending=[False, False, True],
    ).head(10)
    eastern_champion = playoff_probabilities[
        playoff_probabilities["CONFERENCE"].eq("Eastern")
    ].sort_values(["NBA_FINALS_PROB", "CHAMPIONSHIP_PROB", "TEAM"], ascending=[False, False, True]).iloc[0]
    western_champion = playoff_probabilities[
        playoff_probabilities["CONFERENCE"].eq("Western")
    ].sort_values(["NBA_FINALS_PROB", "CHAMPIONSHIP_PROB", "TEAM"], ascending=[False, False, True]).iloc[0]
    nba_champion = top_championships.iloc[0]

    print("\n2026-27 preseason forecast summary")
    print("----------------------------------")
    print(f"Number of teams: {team_count}")
    print("\nTop 10 projected regular-season records:")
    print(
        season_predictions[
            ["TEAM", "PROJECTED_WINS", "PROJECTED_LOSSES", "PROJECTED_WIN_PCT"]
        ].head(10).to_string(index=False)
    )
    print("\nProjected Eastern seeds:")
    print(eastern_seeds.to_string(index=False))
    print("\nProjected Western seeds:")
    print(western_seeds.to_string(index=False))
    print("\nTop 10 championship probabilities:")
    print(
        top_championships[
            ["TEAM", "CONFERENCE", "PROJECTED_SEED", "CHAMPIONSHIP_PROB"]
        ].to_string(index=False)
    )
    print(f"\nMost likely Eastern champion: {eastern_champion['TEAM']}")
    print(f"Most likely Western champion: {western_champion['TEAM']}")
    print(f"Most likely NBA champion: {nba_champion['TEAM']}")
    print(
        "Sum of championship probabilities: "
        f"{playoff_probabilities['CHAMPIONSHIP_PROB'].sum():.6f}"
    )
    print("\nGenerated output files:")
    for path in output_paths.values():
        print(path)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    features, games, model, feature_columns = load_inputs(project_root)
    team_states, _ = build_preseason_team_states(features, games, feature_columns)
    initialize_prediction_context(model, feature_columns, team_states)

    teams = sorted(team_states.index.tolist())

    if len(teams) != 30:
        raise ValueError(f"Expected 30 teams, found {len(teams)}.")

    precompute_matchup_probabilities(teams)

    season_predictions = project_regular_season_records(teams)
    standings, projected_standings = add_conferences_and_seeds(season_predictions)
    playoff_probabilities = simulate_playoffs(standings)
    model_evaluation = load_model_evaluation(data_dir)
    summary = create_summary(standings, playoff_probabilities, model_evaluation)
    output_paths = write_outputs(
        data_dir,
        season_predictions,
        projected_standings,
        playoff_probabilities,
        summary,
    )

    print_forecast_summary(
        len(teams),
        season_predictions,
        projected_standings,
        playoff_probabilities,
        output_paths,
    )


if __name__ == "__main__":
    main()
