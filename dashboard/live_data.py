# Data for the dashboard: loads the live forecast record and the test results from GitHub,
# and turns them into the numbers and tables the charts show. There is no Streamlit code in
# this file, so every calculation can be tested on its own.

import io
import json
from datetime import timedelta
import pandas as pd
import requests

TZ = "Europe/Stockholm"
REPO = "Hossain007Bhuiyan/se3-electricity-forecast"
RAW = f"https://raw.githubusercontent.com/{REPO}"
FORECASTS_URL = f"{RAW}/forecast-data/forecasts.csv"   # written every morning by GitHub Actions
TEST_RESULTS_URL = f"{RAW}/main/results/test_results.csv"
MLFLOW_RUNS_URL = f"{RAW}/main/results/mlflow_runs.csv"           # exported from mlflow.db
MLFLOW_MONTHLY_URL = f"{RAW}/main/results/mlflow_monthly_mae.csv"
MONITORING_URL = f"{RAW}/forecast-data/monitoring.json"  # written every day by the monitoring workflow
RUN_LOG_URL = f"{RAW}/forecast-data/run_log.jsonl"        # one line per daily forecast run
REPO_URL = f"https://github.com/{REPO}"
PRICE_API = "https://www.elprisetjustnu.se/api/v1/prices/{year}/{month:02d}-{day:02d}_SE3.json"


# Reads a CSV file from the internet (or a local path, used by the tests). Web files are downloaded
# with requests, which brings its own up-to-date list of trusted certificates. pandas' own download
# uses Python's built-in one, which on many Macs cannot verify HTTPS certificates.
def read_csv(source):
    if str(source).startswith("http"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    return pd.read_csv(source)


# The live record: one row per forecast hour, with the forecasts and (once published) the real price
def load_record(url=FORECASTS_URL):
    record = read_csv(url)
    record["time_utc"] = pd.to_datetime(record["time_utc"], utc=True)
    record["time_local"] = record["time_utc"].dt.tz_convert(TZ)
    record["issued_at_utc"] = pd.to_datetime(record["issued_at_utc"], utc=True)
    record["issued_before_noon"] = record["issued_before_noon"].astype(str) == "True"
    return record


def load_test_results(url=TEST_RESULTS_URL):
    return read_csv(url)


# The MLflow runs, as exported by "uv run python -m se3_electricity_forecast.tracking"
def load_mlflow_runs(url=MLFLOW_RUNS_URL):
    return read_csv(url)


def load_mlflow_monthly(url=MLFLOW_MONTHLY_URL):
    return read_csv(url)


# The newest forecast day in the record and its rows (usually tomorrow)
def latest_day(record):
    day = record["time_local"].dt.date.max()
    rows = record[record["time_local"].dt.date == day].reset_index(drop=True)
    return day, rows


# Key numbers of one forecast day: average price and the most and least expensive hour,
# all from the LSTM forecast
def day_overview(rows):
    peak = rows.loc[rows["lstm"].idxmax()]
    low = rows.loc[rows["lstm"].idxmin()]
    return {
        "mean": float(rows["lstm"].mean()),
        "peak_price": float(peak["lstm"]),
        "peak_time": peak["time_local"],
        "low_price": float(low["lstm"]),
        "low_time": low["time_local"],
        "hours": int(len(rows)),
    }


# Error per day for the forecasts that count: real price known, and made before 12:00 Swedish time
# on the day before (the same rule as summary.json)
def daily_errors(record):
    done = record[record["actual"].notna() & record["issued_before_noon"]].copy()
    if done.empty:
        return pd.DataFrame(columns=["day", "hours", "lstm_mae", "weekly_naive_mae"])
    done["day"] = done["time_local"].dt.date
    done["lstm_error"] = (done["actual"] - done["lstm"]).abs()
    done["naive_error"] = (done["actual"] - done["weekly_naive"]).abs()
    return (done.groupby("day")
            .agg(hours=("actual", "size"), lstm_mae=("lstm_error", "mean"),
                 weekly_naive_mae=("naive_error", "mean"))
            .reset_index())


# MAE over every window of `days` counted days in a row, weighted by hours, for the LSTM and the
# baseline. Empty until there are at least `days` counted days.
def rolling_errors(errors, days):
    hours = errors["hours"].rolling(days).sum()
    rolling = pd.DataFrame({
        "day": errors["day"],
        "lstm_mae": (errors["lstm_mae"] * errors["hours"]).rolling(days).sum() / hours,
        "weekly_naive_mae": (errors["weekly_naive_mae"] * errors["hours"]).rolling(days).sum() / hours,
    })
    return rolling.dropna().reset_index(drop=True)


# The result of the latest monitoring run (see src/se3_electricity_forecast/monitor.py)
def load_monitoring(url=MONITORING_URL):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()
    
# The run log of the daily forecast, newest run first. Each line of the file is one JSON object.
def load_run_log(url=RUN_LOG_URL):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    runs = pd.DataFrame([json.loads(line) for line in response.text.splitlines() if line.strip()])
    runs["started"] = pd.to_datetime(runs["started_utc"], utc=True).dt.tz_convert(TZ)
    return runs.sort_values("started", ascending=False).reset_index(drop=True)

# Real hourly prices for the last `days` days, straight from the same price API the project uses.
# Since October 2025 the API gives 15-minute prices, so they are averaged per hour, as in the
# project. Days that are not available are skipped.
def fetch_recent_prices(days=30, today=None):
    today = today or pd.Timestamp.now(tz=TZ).date()
    rows = []
    for offset in range(days, -2, -1):  # from `days` days ago up to tomorrow (if already published)
        d = today - timedelta(days=offset)
        response = requests.get(PRICE_API.format(year=d.year, month=d.month, day=d.day), timeout=30)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        rows += response.json()
    return hourly_prices(pd.DataFrame(rows))


def hourly_prices(raw):
    time_utc = pd.to_datetime(raw["time_start"], utc=True).dt.floor("h")
    hourly = raw.assign(time_utc=time_utc).groupby("time_utc")["SEK_per_kWh"].mean().reset_index()
    hourly = hourly.rename(columns={"SEK_per_kWh": "price"})
    hourly["time_local"] = hourly["time_utc"].dt.tz_convert(TZ)
    return hourly


# Prices as a table of days (rows) x hours of the day (columns), for the 3D price landscape.
# On the 25-hour day in October the hour 02:00 exists twice, so the two prices are averaged;
# on the 23-hour day in March 02:00 does not exist and stays empty.
def price_landscape(prices):
    table = prices.assign(day=prices["time_local"].dt.date, hour=prices["time_local"].dt.hour)
    return table.pivot_table(index="day", columns="hour", values="price", aggfunc="mean").reindex(columns=range(24))