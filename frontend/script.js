"use strict";

const DATA_URLS = {
    summary: "../data/forecast_2026_27_summary.json",
    season: "../data/season_2026_27_predictions.csv",
    playoffs: "../data/2027_playoff_probabilities.csv",
    evaluation: "../data/model_evaluation.csv"
};

const API_BASE_URL = globalThis.NBA_API_BASE_URL || "http://localhost:8001";

const SUMMARY_FIELDS = {
    forecastLabel: "forecast_label",
    eastTopSeed: "projected_eastern_conference_1_seed",
    westTopSeed: "projected_western_conference_1_seed",
    topWins: "highest_projected_regular_season_win_total",
    eastChampion: "most_likely_eastern_conference_champion",
    westChampion: "most_likely_western_conference_champion",
    nbaChampion: "most_likely_nba_champion",
    championshipProbability: "championship_probability"
};

const SEASON_FIELDS = [
    "TEAM",
    "PROJECTED_WINS",
    "PROJECTED_LOSSES",
    "PROJECTED_WIN_PCT"
];

const PLAYOFF_FIELDS = [
    "TEAM",
    "CONFERENCE",
    "PROJECTED_SEED",
    "MAKE_PLAYOFFS_PROB",
    "CONF_SEMIFINALS_PROB",
    "CONF_FINALS_PROB",
    "NBA_FINALS_PROB",
    "CHAMPIONSHIP_PROB"
];

const EVALUATION_FIELDS = [
    "MODEL",
    "ACCURACY",
    "ROC_AUC",
    "LOG_LOSS",
    "BRIER_SCORE"
];

const CONFERENCES = ["Eastern", "Western"];

let seasonProjectionRows = [];
let selectedScheduledGame = null;

document.addEventListener("DOMContentLoaded", () => {
    const page = document.body.dataset.page;

    if (page === "home") {
        initHomePage();
    }

    if (page === "season") {
        initSeasonPage();
    }

    if (page === "playoffs") {
        initPlayoffPage();
    }

    if (page === "predictor") {
        initPredictorPage();
    }

    if (page === "methodology") {
        initMethodologyPage();
    }
});

async function initHomePage() {
    const status = document.getElementById("home-status");
    const content = document.getElementById("home-content");

    setStatus(status, "loading", "Loading forecast data...");

    try {
        const [summary, playoffs] = await Promise.all([
            loadJson(DATA_URLS.summary),
            loadCsv(DATA_URLS.playoffs)
        ]);

        requireFields(summary, Object.values(SUMMARY_FIELDS), "forecast summary");
        validateRows(playoffs, PLAYOFF_FIELDS, "playoff probabilities");

        renderHomeSummary(summary);
        renderContenders(playoffs, "contender-grid", 4);

        setStatus(status, "success", "Forecast data loaded.");
        showElement(content);
    } catch (error) {
        setStatus(status, "error", formatLoadError(error, "forecast data"));
    }
}

async function initSeasonPage() {
    const status = document.getElementById("season-status");
    const content = document.getElementById("season-content");

    setStatus(status, "loading", "Loading season projections...");

    try {
        const [season, playoffs] = await Promise.all([
            loadCsv(DATA_URLS.season),
            loadCsv(DATA_URLS.playoffs)
        ]);

        validateRows(season, SEASON_FIELDS, "season projections");
        validateRows(playoffs, PLAYOFF_FIELDS, "playoff probabilities");
        seasonProjectionRows = enrichSeasonRows(season, playoffs);
        validateConferenceCounts(seasonProjectionRows, "season projections");

        renderSeasonConferenceTables(seasonProjectionRows, "wins");
        setupSeasonSorting();

        setStatus(status, "success", `Loaded season projections for ${seasonProjectionRows.length} teams.`);
        showElement(content);
    } catch (error) {
        setStatus(status, "error", formatLoadError(error, "season projections"));
    }
}

async function initPlayoffPage() {
    const status = document.getElementById("playoff-status");
    const content = document.getElementById("playoff-content");

    setStatus(status, "loading", "Loading playoff probabilities...");

    try {
        const playoffs = await loadCsv(DATA_URLS.playoffs);
        validateRows(playoffs, PLAYOFF_FIELDS, "playoff probabilities");
        validateConferenceCounts(
            playoffs.map((row) => ({
                conference: row.CONFERENCE,
                team: row.TEAM
            })),
            "playoff probabilities"
        );

        renderContenders(playoffs, "playoff-leader-grid", 6);
        renderPlayoffConferenceTables(playoffs);

        setStatus(status, "success", `Loaded playoff probabilities for ${playoffs.length} teams.`);
        showElement(content);
    } catch (error) {
        setStatus(status, "error", formatLoadError(error, "playoff probabilities"));
    }
}

async function initMethodologyPage() {
    const status = document.getElementById("methodology-status");
    const content = document.getElementById("methodology-content");

    setStatus(status, "loading", "Loading model metrics...");

    try {
        const evaluation = await loadCsv(DATA_URLS.evaluation);
        validateRows(evaluation, EVALUATION_FIELDS, "model evaluation");
        renderModelMetrics(evaluation);

        setStatus(status, "success", "Model metrics loaded.");
        showElement(content);
    } catch (error) {
        setStatus(status, "error", formatLoadError(error, "model metrics"));
    }
}

async function initPredictorPage() {
    const status = document.getElementById("predictor-status");
    const content = document.getElementById("predictor-content");
    const gameSearchForm = document.getElementById("game-search-form");
    const dateFilter = document.getElementById("game-date-filter");
    const teamFilter = document.getElementById("game-team-filter");
    const gameSearchButton = document.getElementById("game-search-button");
    const gameSearchError = document.getElementById("game-search-error");
    const gameList = document.getElementById("game-list");
    const gameResultsSummary = document.getElementById("game-results-summary");
    const predictGameButton = document.getElementById("predict-game-button");
    const form = document.getElementById("predictor-form");
    const homeSelect = document.getElementById("home-team-select");
    const awaySelect = document.getElementById("away-team-select");
    const predictButton = document.getElementById("predict-button");
    const errorMessage = document.getElementById("predictor-error");

    setStatus(status, "loading", "Loading predictor...");

    try {
        const teams = await loadTeams();

        if (teams.length !== 30) {
            throw new Error(`Expected 30 teams, found ${teams.length}.`);
        }

        populateTeamSelect(teamFilter, teams, "Any team");
        populateTeamSelect(homeSelect, teams);
        populateTeamSelect(awaySelect, teams);
        setDefaultMatchup(homeSelect, awaySelect, teams);
        updatePredictorControls(homeSelect, awaySelect, predictButton, errorMessage);
        resetScheduledGames(gameList, gameResultsSummary, predictGameButton);

        gameSearchForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            await submitGameSearch({
                date: dateFilter.value,
                team: teamFilter.value,
                gameList,
                gameResultsSummary,
                predictGameButton,
                gameSearchButton,
                gameSearchError,
                status
            });
        });

        [dateFilter, teamFilter].forEach((field) => {
            field.addEventListener("change", () => {
                setFormMessage(gameSearchError, "");
                resetScheduledGames(gameList, gameResultsSummary, predictGameButton);
                hideElement(document.getElementById("scheduled-prediction-result"));
            });
        });

        predictGameButton.addEventListener("click", async () => {
            await submitScheduledPrediction(predictGameButton, gameSearchError, status);
        });

        [homeSelect, awaySelect].forEach((select) => {
            select.addEventListener("change", () => {
                updatePredictorControls(homeSelect, awaySelect, predictButton, errorMessage);
            });
        });

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            await submitPrediction(homeSelect, awaySelect, predictButton, errorMessage, status);
        });

        setStatus(status, "success", "Predictor ready.");
        showElement(content);
    } catch (error) {
        setStatus(status, "error", formatLoadError(error, "predictor"));
    }
}

async function loadJson(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
    }

    return response.json();
}

async function loadText(url) {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}.`);
    }

    return response.text();
}

async function loadCsv(url) {
    const text = await loadText(url);
    return parseCsv(text);
}

async function loadTeams() {
    const response = await fetch(`${API_BASE_URL}/api/teams`);

    if (!response.ok) {
        throw new Error(await apiErrorMessage(response));
    }

    return response.json();
}

async function loadGames(filters = {}) {
    const params = new URLSearchParams();

    if (filters.date) {
        params.set("date", filters.date);
    }

    if (filters.team) {
        params.set("team", filters.team);
    }

    const query = params.toString();
    const response = await fetch(`${API_BASE_URL}/api/games${query ? `?${query}` : ""}`);

    if (!response.ok) {
        throw new Error(await apiErrorMessage(response));
    }

    return response.json();
}

async function getScheduledPrediction(gameId) {
    const response = await fetch(`${API_BASE_URL}/api/predict-game/${encodeURIComponent(gameId)}`);

    if (!response.ok) {
        throw new Error(await apiErrorMessage(response));
    }

    return response.json();
}

async function postMatchupPrediction(homeTeam, awayTeam) {
    const response = await fetch(`${API_BASE_URL}/api/predict-matchup`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            home_team: homeTeam,
            away_team: awayTeam
        })
    });

    if (!response.ok) {
        throw new Error(await apiErrorMessage(response));
    }

    return response.json();
}

async function apiErrorMessage(response) {
    let detail = `Request failed with status ${response.status}.`;

    try {
        const errorBody = await response.json();
        detail = errorBody.detail || detail;
    } catch (error) {
        detail = `Request failed with status ${response.status}.`;
    }

    return detail;
}

function validateRows(rows, fields, label) {
    if (!rows.length) {
        throw new Error(`${label} did not contain any rows.`);
    }

    requireFields(rows[0], fields, label);
}

function requireFields(record, fields, label) {
    const missing = fields.filter((field) => !(field in record));

    if (missing.length) {
        throw new Error(`${label} is missing: ${missing.join(", ")}.`);
    }
}

function validateConferenceCounts(rows, label) {
    CONFERENCES.forEach((conference) => {
        const count = rows.filter((row) => row.conference === conference).length;

        if (count !== 15) {
            throw new Error(`${label} expected 15 ${conference} teams, found ${count}.`);
        }
    });
}

function enrichSeasonRows(season, playoffs) {
    const playoffByTeam = new Map(playoffs.map((row) => [row.TEAM, row]));

    return season.map((row) => {
        const playoffRow = playoffByTeam.get(row.TEAM);

        if (!playoffRow) {
            throw new Error(`No playoff probability row found for ${row.TEAM}.`);
        }

        return {
            team: row.TEAM,
            conference: playoffRow.CONFERENCE,
            seed: Number(playoffRow.PROJECTED_SEED),
            projectedWins: Number(row.PROJECTED_WINS),
            projectedLosses: Number(row.PROJECTED_LOSSES),
            projectedWinPct: Number(row.PROJECTED_WIN_PCT)
        };
    });
}

function renderHomeSummary(summary) {
    const champion = summary[SUMMARY_FIELDS.nbaChampion];
    const championProbability = Number(summary[SUMMARY_FIELDS.championshipProbability]);
    const topWins = summary[SUMMARY_FIELDS.topWins];

    setText("champion-name", champion);
    setText(
        "champion-summary",
        `${champion} has the strongest title odds in the ${summary[SUMMARY_FIELDS.forecastLabel]}.`
    );
    setText("champion-probability", formatProbability(championProbability, 2));
    setText("eastern-champion", summary[SUMMARY_FIELDS.eastChampion]);
    setText("western-champion", summary[SUMMARY_FIELDS.westChampion]);
    setText("east-top-seed", summary[SUMMARY_FIELDS.eastTopSeed]);
    setText("west-top-seed", summary[SUMMARY_FIELDS.westTopSeed]);
    setText("top-win-team", topWins.team);
    setText("top-win-total", String(topWins.projected_wins));
}

function populateTeamSelect(select, teams, emptyLabel = "") {
    const options = teams.map((team) => {
        const option = createElement("option", "", team);
        option.value = team;
        return option;
    });

    if (emptyLabel) {
        const emptyOption = createElement("option", "", emptyLabel);
        emptyOption.value = "";
        options.unshift(emptyOption);
    }

    select.replaceChildren(...options);
}

function setDefaultMatchup(homeSelect, awaySelect, teams) {
    homeSelect.value = teams.includes("BOS") ? "BOS" : teams[0];
    awaySelect.value = teams.includes("LAL") ? "LAL" : teams.find((team) => team !== homeSelect.value);
}

function updatePredictorControls(homeSelect, awaySelect, predictButton, errorMessage) {
    const hasIdenticalTeams = homeSelect.value === awaySelect.value;

    for (const option of homeSelect.options) {
        option.disabled = option.value === awaySelect.value && option.value !== homeSelect.value;
    }

    for (const option of awaySelect.options) {
        option.disabled = option.value === homeSelect.value && option.value !== awaySelect.value;
    }

    predictButton.disabled = hasIdenticalTeams;
    setFormMessage(
        errorMessage,
        hasIdenticalTeams ? "Choose two different teams." : ""
    );
}

async function submitPrediction(homeSelect, awaySelect, predictButton, errorMessage, status) {
    if (homeSelect.value === awaySelect.value) {
        setFormMessage(errorMessage, "Choose two different teams.");
        return;
    }

    predictButton.disabled = true;
    setFormMessage(errorMessage, "");
    setStatus(status, "loading", "Predicting matchup...");
    let failureMessage = "";

    try {
        const prediction = await postMatchupPrediction(homeSelect.value, awaySelect.value);
        renderPredictionResult(prediction);
        setStatus(status, "success", "Prediction ready.");
    } catch (error) {
        failureMessage = error.message;
        setStatus(status, "error", "Prediction failed.");
        setFormMessage(errorMessage, error.message);
    } finally {
        updatePredictorControls(homeSelect, awaySelect, predictButton, errorMessage);

        if (failureMessage) {
            setFormMessage(errorMessage, failureMessage);
        }
    }
}

async function submitGameSearch({
    date,
    team,
    gameList,
    gameResultsSummary,
    predictGameButton,
    gameSearchButton,
    gameSearchError,
    status
}) {
    const hasFilter = Boolean(date || team);

    if (!hasFilter) {
        resetScheduledGames(gameList, gameResultsSummary, predictGameButton);
        setFormMessage(gameSearchError, "Choose a date or team before searching.");
        return;
    }

    gameSearchButton.disabled = true;
    predictGameButton.disabled = true;
    selectedScheduledGame = null;
    setFormMessage(gameSearchError, "");
    setStatus(status, "loading", "Loading scheduled games...");
    hideElement(document.getElementById("scheduled-prediction-result"));

    try {
        const games = await loadGames({ date, team });
        renderScheduledGames(games, gameList, gameResultsSummary, predictGameButton);
        setStatus(
            status,
            "success",
            games.length === 1 ? "Loaded 1 scheduled game." : `Loaded ${games.length} scheduled games.`
        );
    } catch (error) {
        resetScheduledGames(gameList, gameResultsSummary, predictGameButton);
        setStatus(status, "error", "Unable to load scheduled games.");
        setFormMessage(gameSearchError, error.message);
    } finally {
        gameSearchButton.disabled = false;
    }
}

async function submitScheduledPrediction(predictGameButton, gameSearchError, status) {
    if (!selectedScheduledGame) {
        setFormMessage(gameSearchError, "Select a scheduled game first.");
        return;
    }

    predictGameButton.disabled = true;
    setFormMessage(gameSearchError, "");
    setStatus(status, "loading", "Predicting scheduled game...");

    try {
        const prediction = await getScheduledPrediction(selectedScheduledGame.game_id);
        renderScheduledPrediction(prediction);
        setStatus(status, "success", "Scheduled game prediction ready.");
    } catch (error) {
        setStatus(status, "error", "Prediction failed.");
        setFormMessage(gameSearchError, error.message);
    } finally {
        predictGameButton.disabled = !selectedScheduledGame;
    }
}

function resetScheduledGames(gameList, gameResultsSummary, predictGameButton) {
    selectedScheduledGame = null;

    if (gameResultsSummary) {
        gameResultsSummary.textContent = "Choose a filter to load matching games.";
    }

    if (gameList) {
        gameList.replaceChildren(createElement("p", "empty-state", "No games loaded yet."));
    }

    if (predictGameButton) {
        predictGameButton.disabled = true;
    }
}

function renderScheduledGames(games, gameList, gameResultsSummary, predictGameButton) {
    selectedScheduledGame = null;
    predictGameButton.disabled = true;

    if (!games.length) {
        gameResultsSummary.textContent = "No games found for those filters.";
        gameList.replaceChildren(createElement("p", "empty-state", "Try a different date or team."));
        return;
    }

    gameResultsSummary.textContent = games.length === 1
        ? "1 game found. Select it to continue."
        : `${games.length} games found. Select one to continue.`;

    gameList.replaceChildren(
        ...games.map((game) => createGameOption(game, predictGameButton))
    );
}

function createGameOption(game, predictGameButton) {
    const button = document.createElement("button");
    const meta = createElement(
        "span",
        "game-option-meta",
        `${formatDateLabel(game.game_date)} | ${formatGameTime(game.game_time)}`
    );
    const matchup = createElement(
        "strong",
        "",
        `${game.away_team} @ ${game.home_team}`
    );
    const location = createElement("span", "", formatLocation(game));

    button.type = "button";
    button.className = "game-option";
    button.dataset.gameId = game.game_id;
    button.setAttribute("aria-pressed", "false");
    button.append(meta, matchup, location);

    button.addEventListener("click", () => {
        selectedScheduledGame = game;

        document.querySelectorAll(".game-option").forEach((option) => {
            option.classList.toggle("is-selected", option === button);
            option.setAttribute("aria-pressed", String(option === button));
        });

        predictGameButton.disabled = false;
    });

    return button;
}

function renderScheduledPrediction(prediction) {
    const result = document.getElementById("scheduled-prediction-result");
    const homeProbability = Number(prediction.home_win_probability);
    const awayProbability = Number(prediction.away_win_probability);

    setText("scheduled-matchup", `${prediction.away_team} @ ${prediction.home_team}`);
    setText("scheduled-game-date", formatDateLabel(prediction.game_date));
    setText("scheduled-game-time", formatGameTime(prediction.game_time));
    setText("scheduled-game-location", formatLocation(prediction));
    setText("scheduled-prediction-winner", `${prediction.predicted_winner} is favored`);
    setText("scheduled-prediction-confidence", `${prediction.confidence} confidence`);
    setText("scheduled-home-result-team", prediction.home_team);
    setText("scheduled-away-result-team", prediction.away_team);
    setText("scheduled-home-result-probability", formatProbability(homeProbability, 1));
    setText("scheduled-away-result-probability", formatProbability(awayProbability, 1));

    setProbabilityBar("scheduled-home-result-bar", homeProbability);
    setProbabilityBar("scheduled-away-result-bar", awayProbability);
    showElement(result);
}

function renderPredictionResult(prediction) {
    const result = document.getElementById("prediction-result");
    const homeProbability = Number(prediction.home_win_probability);
    const awayProbability = Number(prediction.away_win_probability);

    setText("prediction-winner", `${prediction.predicted_winner} is favored`);
    setText("prediction-confidence", `${prediction.confidence} confidence`);
    setText("home-result-team", prediction.home_team);
    setText("away-result-team", prediction.away_team);
    setText("home-result-probability", formatProbability(homeProbability, 1));
    setText("away-result-probability", formatProbability(awayProbability, 1));

    setProbabilityBar("home-result-bar", homeProbability);
    setProbabilityBar("away-result-bar", awayProbability);
    showElement(result);
}

function renderContenders(playoffs, containerId, limit) {
    const container = document.getElementById(containerId);

    if (!container) {
        return;
    }

    const sorted = [...playoffs].sort(compareByChampionship).slice(0, limit);
    container.replaceChildren(
        ...sorted.map((row, index) => createContenderCard(row, index + 1))
    );
}

function createContenderCard(row, rank) {
    const championshipProbability = Number(row.CHAMPIONSHIP_PROB);
    const card = createElement("article", "contender-card");
    const rankLabel = createElement("span", "rank-label", `#${rank}`);
    const title = createElement("h3", "", row.TEAM);
    const meta = createElement("p", "", `${row.CONFERENCE} seed ${row.PROJECTED_SEED}`);
    const probability = createElement(
        "strong",
        "probability-value",
        formatProbability(championshipProbability, 2)
    );
    const meter = createProbabilityMeter(championshipProbability, "probability-meter");

    card.append(rankLabel, title, meta, probability, meter);
    return card;
}

function setupSeasonSorting() {
    document.querySelectorAll("[data-season-sort]").forEach((button) => {
        button.addEventListener("click", () => {
            const sortMode = button.dataset.seasonSort;

            renderSeasonConferenceTables(seasonProjectionRows, sortMode);

            document.querySelectorAll("[data-season-sort]").forEach((control) => {
                control.setAttribute(
                    "aria-pressed",
                    String(control.dataset.seasonSort === sortMode)
                );
            });
        });
    });
}

function renderSeasonConferenceTables(rows, sortMode) {
    renderSeasonConference(rows, "Eastern", "season-eastern-body", sortMode);
    renderSeasonConference(rows, "Western", "season-western-body", sortMode);
}

function renderSeasonConference(rows, conference, bodyId, sortMode) {
    const body = document.getElementById(bodyId);

    if (!body) {
        return;
    }

    const sortedRows = sortSeasonRows(
        rows.filter((row) => row.conference === conference),
        sortMode
    );

    body.replaceChildren(...sortedRows.map(createSeasonRow));
}

function sortSeasonRows(rows, sortMode) {
    const sorted = [...rows];

    if (sortMode === "team") {
        return sorted.sort((a, b) => a.team.localeCompare(b.team));
    }

    return sorted.sort((a, b) =>
        b.projectedWins - a.projectedWins
        || b.projectedWinPct - a.projectedWinPct
        || a.team.localeCompare(b.team)
    );
}

function createSeasonRow(row) {
    const tableRow = document.createElement("tr");
    const teamHeader = document.createElement("th");

    teamHeader.scope = "row";
    teamHeader.textContent = row.team;

    tableRow.append(
        createTableCell(formatSeed(row.seed)),
        teamHeader,
        createTableCell(`${row.projectedWins}-${row.projectedLosses}`),
        createTableCell(formatWinPct(row.projectedWinPct))
    );

    return tableRow;
}

function renderPlayoffConferenceTables(playoffs) {
    renderPlayoffConference(playoffs, "Eastern", "playoff-eastern-body");
    renderPlayoffConference(playoffs, "Western", "playoff-western-body");
}

function renderPlayoffConference(playoffs, conference, bodyId) {
    const body = document.getElementById(bodyId);

    if (!body) {
        return;
    }

    const rows = playoffs
        .filter((row) => row.CONFERENCE === conference)
        .sort((a, b) => Number(a.PROJECTED_SEED) - Number(b.PROJECTED_SEED));

    body.replaceChildren(...rows.map(createPlayoffRow));
}

function createPlayoffRow(row) {
    const tableRow = document.createElement("tr");
    const teamHeader = document.createElement("th");

    teamHeader.scope = "row";
    teamHeader.textContent = row.TEAM;

    tableRow.append(
        createTableCell(formatSeed(Number(row.PROJECTED_SEED))),
        teamHeader,
        createTableCell(formatProbability(row.MAKE_PLAYOFFS_PROB, 1)),
        createTableCell(formatProbability(row.CONF_SEMIFINALS_PROB, 1)),
        createTableCell(formatProbability(row.CONF_FINALS_PROB, 1)),
        createTableCell(formatProbability(row.NBA_FINALS_PROB, 1)),
        createChampionshipCell(row.CHAMPIONSHIP_PROB)
    );

    return tableRow;
}

function createChampionshipCell(value) {
    const cell = document.createElement("td");
    const wrapper = createElement("div", "championship-cell");
    const text = createElement("span", "", formatProbability(value, 2));
    const meter = createProbabilityMeter(value, "mini-meter");

    wrapper.append(text, meter);
    cell.append(wrapper);
    return cell;
}

function createProbabilityMeter(value, className) {
    const meter = createElement("div", className);
    const fill = createElement("span");

    fill.style.setProperty("--probability", clampProbability(value));
    meter.append(fill);
    return meter;
}

function setProbabilityBar(id, value) {
    const bar = document.getElementById(id);

    if (bar) {
        bar.style.setProperty("--probability", clampProbability(value));
    }
}

function renderModelMetrics(evaluation) {
    const metricGrid = document.getElementById("metric-grid");

    if (!metricGrid) {
        return;
    }

    const logisticRegression = evaluation.find((row) => row.MODEL === "Logistic Regression");

    if (!logisticRegression) {
        throw new Error("Model evaluation is missing Logistic Regression results.");
    }

    const metrics = [
        ["Accuracy", formatProbability(logisticRegression.ACCURACY, 1)],
        ["ROC AUC", formatNumber(logisticRegression.ROC_AUC, 3)],
        ["Log Loss", formatNumber(logisticRegression.LOG_LOSS, 3)],
        ["Brier Score", formatNumber(logisticRegression.BRIER_SCORE, 3)]
    ];

    metricGrid.replaceChildren(
        ...metrics.map(([label, value]) => {
            const card = createElement("article", "metric-card");
            card.append(
                createElement("span", "mini-label", label),
                createElement("strong", "", value)
            );
            return card;
        })
    );
}

function compareByChampionship(a, b) {
    return Number(b.CHAMPIONSHIP_PROB) - Number(a.CHAMPIONSHIP_PROB)
        || Number(b.NBA_FINALS_PROB) - Number(a.NBA_FINALS_PROB)
        || a.TEAM.localeCompare(b.TEAM);
}

function createTableCell(text) {
    const cell = document.createElement("td");
    cell.textContent = text;
    return cell;
}

// Small CSV parser so the static app does not need build tooling or libraries.
function parseCsv(text) {
    const rows = [];
    let row = [];
    let value = "";
    let inQuotes = false;

    for (let index = 0; index < text.length; index += 1) {
        const character = text[index];
        const nextCharacter = text[index + 1];

        if (character === '"') {
            if (inQuotes && nextCharacter === '"') {
                value += '"';
                index += 1;
            } else {
                inQuotes = !inQuotes;
            }
            continue;
        }

        if (character === "," && !inQuotes) {
            row.push(value);
            value = "";
            continue;
        }

        if ((character === "\n" || character === "\r") && !inQuotes) {
            if (character === "\r" && nextCharacter === "\n") {
                index += 1;
            }

            row.push(value);
            rows.push(row);
            row = [];
            value = "";
            continue;
        }

        value += character;
    }

    if (value || row.length) {
        row.push(value);
        rows.push(row);
    }

    const [headers = [], ...bodyRows] = rows.filter((cells) =>
        cells.some((cell) => cell.trim() !== "")
    );

    return bodyRows.map((cells) => {
        const record = {};

        headers.forEach((header, index) => {
            record[header] = cells[index] ?? "";
        });

        return record;
    });
}

function setText(id, value) {
    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}

function setStatus(element, type, message) {
    if (!element) {
        return;
    }

    element.className = `${type}-message`;
    element.textContent = message;
}

function setFormMessage(element, message) {
    if (element) {
        element.textContent = message;
    }
}

function showElement(element) {
    if (element) {
        element.classList.remove("is-hidden");
    }
}

function hideElement(element) {
    if (element) {
        element.classList.add("is-hidden");
    }
}

function formatLoadError(error, label) {
    return `Unable to load ${label}. ${error.message}`;
}

function formatProbability(value, decimals = 1) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "--";
    }

    const percent = Math.abs(numericValue) <= 1 ? numericValue * 100 : numericValue;
    return `${percent.toFixed(decimals)}%`;
}

function clampProbability(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "0%";
    }

    const percent = Math.abs(numericValue) <= 1 ? numericValue * 100 : numericValue;
    return `${Math.max(0, Math.min(100, percent))}%`;
}

function formatWinPct(value) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "--";
    }

    return numericValue.toFixed(3).replace(/^0/, "");
}

function formatNumber(value, decimals) {
    const numericValue = Number(value);

    if (!Number.isFinite(numericValue)) {
        return "--";
    }

    return numericValue.toFixed(decimals);
}

function formatDateLabel(value) {
    if (!value) {
        return "Date TBA";
    }

    const date = new Date(`${value}T00:00:00`);

    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric"
    });
}

function formatGameTime(value) {
    if (!value) {
        return "Time TBA";
    }

    const time = String(value).trim();

    if (!time || time.toLowerCase() === "tbd") {
        return "Time TBA";
    }

    const normalized = time.replace(/\b(am|pm|et|ct|mt|pt)\b/gi, (match) =>
        match.toUpperCase()
    );

    if (/\b(ET|CT|MT|PT)\b/.test(normalized)) {
        return normalized;
    }

    return `${normalized} ET`;
}

function formatLocation(record) {
    const arena = record.arena || "Arena TBA";
    const city = record.city || "";
    const state = record.state_or_country || "";
    const cityState = [city, state].filter(Boolean).join(", ");

    return cityState ? `${arena} - ${cityState}` : arena;
}

function formatSeed(seed) {
    if (!Number.isFinite(seed)) {
        return "--";
    }

    if (seed <= 6) {
        return `${seed}`;
    }

    if (seed <= 10) {
        return `${seed} Play-In`;
    }

    return `${seed} Out`;
}

function createElement(tagName, className = "", text = "") {
    const element = document.createElement(tagName);

    if (className) {
        element.className = className;
    }

    if (text) {
        element.textContent = text;
    }

    return element;
}
