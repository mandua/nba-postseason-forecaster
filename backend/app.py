from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "nba_win_model.joblib"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"
GAMES_PATH = DATA_DIR / "nba_games_clean.csv"
SCHEDULE_PATH = DATA_DIR / "nba_schedule_2026_27.csv"

SCHEDULE_COLUMNS = [
    "GAME_ID",
    "GAME_DATE",
    "GAME_TIME",
    "HOME_TEAM",
    "AWAY_TEAM",
    "ARENA",
    "CITY",
    "STATE_OR_COUNTRY",
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_2026_27_forecast import (  # noqa: E402
    calculate_completed_season_team_states,
    create_completed_team_history,
    get_base_feature_names,
    validate_clean_games,
)


class MatchupRequest(BaseModel):
    home_team: str
    away_team: str


class MatchupResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str
    confidence: str


class ScheduledGame(BaseModel):
    game_id: str
    game_date: str
    game_time: str
    home_team: str
    away_team: str
    arena: str
    city: str
    state_or_country: str


class ScheduledGamePrediction(ScheduledGame, MatchupResponse):
    pass


class PredictorState:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        self.base_feature_names = get_base_feature_names(self.feature_columns)
        self.team_states = self._build_team_states()
        self.teams = sorted(self.team_states.index.tolist())

        if len(self.teams) != 30:
            raise ValueError(f"Expected 30 NBA teams, found {len(self.teams)}.")

    def _build_team_states(self):
        games = pd.read_csv(GAMES_PATH, dtype={"GAME_ID": str})
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="raise")
        games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        validate_clean_games(games)
        team_history = create_completed_team_history(games)
        return calculate_completed_season_team_states(
            team_history,
            self.base_feature_names,
        )

    def build_model_row(
        self,
        home_team: str,
        away_team: str,
        feature_overrides: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        row = {}
        feature_overrides = feature_overrides or {}

        for column in self.feature_columns:
            if column.startswith("HOME_"):
                side = "HOME"
                team = home_team
                feature_name = column.removeprefix("HOME_")
            elif column.startswith("AWAY_"):
                side = "AWAY"
                team = away_team
                feature_name = column.removeprefix("AWAY_")
            else:
                raise ValueError(f"Unexpected feature column: {column}")

            row[column] = feature_overrides.get(side, {}).get(
                feature_name,
                self.team_states.loc[team, feature_name],
            )

        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(
        self,
        home_team: str,
        away_team: str,
        feature_overrides: dict[str, dict[str, float]] | None = None,
    ) -> MatchupResponse:
        model_row = self.build_model_row(home_team, away_team, feature_overrides)
        home_probability = float(self.model.predict_proba(model_row)[0, 1])
        away_probability = 1.0 - home_probability

        if not np.isclose(home_probability + away_probability, 1.0, atol=1e-9):
            raise ValueError("Home and away probabilities do not sum to 1.")

        winning_probability = max(home_probability, away_probability)
        predicted_winner = home_team if home_probability >= away_probability else away_team

        return MatchupResponse(
            home_team=home_team,
            away_team=away_team,
            home_win_probability=home_probability,
            away_win_probability=away_probability,
            predicted_winner=predicted_winner,
            confidence=confidence_label(winning_probability),
        )


def confidence_label(winning_probability: float) -> str:
    if winning_probability >= 0.65:
        return "High"

    if winning_probability >= 0.55:
        return "Medium"

    return "Low"


@lru_cache(maxsize=1)
def get_predictor_state() -> PredictorState:
    return PredictorState()


@lru_cache(maxsize=1)
def get_schedule_frame() -> pd.DataFrame:
    if not SCHEDULE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Schedule file is unavailable. Run "
                "scripts/download_2026_27_schedule.py first."
            ),
        )

    schedule = pd.read_csv(SCHEDULE_PATH, dtype={"GAME_ID": str}, keep_default_na=False)
    missing_columns = sorted(set(SCHEDULE_COLUMNS) - set(schedule.columns))

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Schedule file is missing columns: {', '.join(missing_columns)}.",
        )

    for column in SCHEDULE_COLUMNS:
        schedule[column] = schedule[column].astype(str).str.strip()

    if schedule["GAME_ID"].duplicated().any():
        raise HTTPException(status_code=500, detail="Schedule contains duplicate GAME_ID values.")

    if schedule[["GAME_ID", "GAME_DATE", "HOME_TEAM", "AWAY_TEAM"]].eq("").any().any():
        raise HTTPException(status_code=500, detail="Schedule has missing game IDs, dates, or teams.")

    if schedule["HOME_TEAM"].eq(schedule["AWAY_TEAM"]).any():
        raise HTTPException(status_code=500, detail="Schedule contains identical home and away teams.")

    schedule["_GAME_DATE"] = pd.to_datetime(schedule["GAME_DATE"], errors="raise").dt.normalize()

    if "GAME_DATETIME_UTC" in schedule.columns:
        schedule["_GAME_DATETIME"] = pd.to_datetime(
            schedule["GAME_DATETIME_UTC"].replace("", pd.NA),
            errors="coerce",
            utc=True,
        )
    else:
        schedule["_GAME_DATETIME"] = pd.NaT

    fallback_datetime = pd.to_datetime(schedule["GAME_DATE"], errors="raise", utc=True)
    schedule["_GAME_DATETIME"] = schedule["_GAME_DATETIME"].fillna(fallback_datetime)

    schedule = schedule.sort_values(["_GAME_DATETIME", "GAME_ID"]).reset_index(drop=True)
    teams = set(schedule["HOME_TEAM"]) | set(schedule["AWAY_TEAM"])

    if len(teams) != 30:
        raise HTTPException(status_code=500, detail=f"Schedule contains {len(teams)} teams, expected 30.")

    return schedule


def normalize_team(team: str) -> str:
    return team.strip().upper()


def validate_matchup(home_team: str, away_team: str, teams: set[str]) -> tuple[str, str]:
    home_team = normalize_team(home_team)
    away_team = normalize_team(away_team)
    unknown_teams = sorted({team for team in [home_team, away_team] if team not in teams})

    if unknown_teams:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown team abbreviation(s): {', '.join(unknown_teams)}.",
        )

    if home_team == away_team:
        raise HTTPException(
            status_code=400,
            detail="Home team and away team must be different.",
        )

    return home_team, away_team


def validate_team_filter(team: str, teams: set[str]) -> str:
    team = normalize_team(team)

    if team not in teams:
        raise HTTPException(status_code=400, detail=f"Unknown team abbreviation: {team}.")

    return team


def parse_date_filter(date_value: str) -> str:
    try:
        return pd.to_datetime(
            date_value,
            format="%Y-%m-%d",
            errors="raise",
        ).strftime("%Y-%m-%d")
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Date must use YYYY-MM-DD format.") from error


def game_record(row: pd.Series) -> ScheduledGame:
    return ScheduledGame(
        game_id=row["GAME_ID"],
        game_date=row["GAME_DATE"],
        game_time=row["GAME_TIME"],
        home_team=row["HOME_TEAM"],
        away_team=row["AWAY_TEAM"],
        arena=row["ARENA"],
        city=row["CITY"],
        state_or_country=row["STATE_OR_COUNTRY"],
    )


def model_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def calculate_rest_for_team(
    schedule: pd.DataFrame,
    row: pd.Series,
    team: str,
    state: PredictorState,
) -> dict[str, float]:
    team_games = schedule[
        schedule["HOME_TEAM"].eq(team) | schedule["AWAY_TEAM"].eq(team)
    ].sort_values(["_GAME_DATETIME", "GAME_ID"])
    previous_games = team_games[team_games["_GAME_DATETIME"] < row["_GAME_DATETIME"]]

    if previous_games.empty:
        return {
            "REST_DAYS": float(state.team_states.loc[team, "REST_DAYS"]),
            "BACK_TO_BACK": 0.0,
        }

    previous_game = previous_games.iloc[-1]
    rest_days = int((row["_GAME_DATE"] - previous_game["_GAME_DATE"]).days)

    return {
        "REST_DAYS": float(max(rest_days, 0)),
        "BACK_TO_BACK": 1.0 if rest_days == 1 else 0.0,
    }


def scheduled_feature_overrides(schedule: pd.DataFrame, row: pd.Series, state: PredictorState):
    return {
        "HOME": calculate_rest_for_team(schedule, row, row["HOME_TEAM"], state),
        "AWAY": calculate_rest_for_team(schedule, row, row["AWAY_TEAM"], state),
    }


app = FastAPI(title="NBA Matchup Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/teams", response_model=list[str])
def get_teams():
    return get_predictor_state().teams


@app.post("/api/predict-matchup", response_model=MatchupResponse)
def predict_matchup(request: MatchupRequest):
    state = get_predictor_state()
    home_team, away_team = validate_matchup(
        request.home_team,
        request.away_team,
        set(state.teams),
    )

    try:
        return state.predict(home_team, away_team)
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/games", response_model=list[ScheduledGame])
def get_games(date: str | None = None, team: str | None = None):
    state = get_predictor_state()
    schedule = get_schedule_frame()
    filtered = schedule

    if date:
        filtered = filtered[filtered["GAME_DATE"].eq(parse_date_filter(date))]

    if team:
        team = validate_team_filter(team, set(state.teams))
        filtered = filtered[
            filtered["HOME_TEAM"].eq(team) | filtered["AWAY_TEAM"].eq(team)
        ]

    return [game_record(row) for _, row in filtered.iterrows()]


@app.get("/api/predict-game/{game_id}", response_model=ScheduledGamePrediction)
def predict_game(game_id: str):
    state = get_predictor_state()
    schedule = get_schedule_frame()
    game_id = game_id.strip()
    matching_rows = schedule[schedule["GAME_ID"].eq(game_id)]

    if matching_rows.empty:
        raise HTTPException(status_code=404, detail=f"Scheduled game not found: {game_id}.")

    game = matching_rows.iloc[0]
    home_team, away_team = validate_matchup(
        game["HOME_TEAM"],
        game["AWAY_TEAM"],
        set(state.teams),
    )
    overrides = scheduled_feature_overrides(schedule, game, state)

    try:
        prediction = state.predict(home_team, away_team, overrides)
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    response = model_dict(game_record(game))
    response.update(model_dict(prediction))
    return ScheduledGamePrediction(**response)
