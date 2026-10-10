# Monitoring of the live forecasts. Three checks run every day after 12:00 Swedish time:
#   1. deadline: tomorrow's forecast was made before 12:00, when bids for the day-ahead market close
#   2. error: the live error of the last 7 counted days is not above a fixed limit
#   3. drift: the model inputs of the last 14 days look like the inputs of the same month in the
#      test year (prices and weather can change so much that the model no longer fits them)
#
# The limits come from the test year and are saved once in results/monitoring_reference.json, so
# they are fixed in advance and cannot be changed later to hide a bad result. Make that file with:
#     uv run python -m se3_electricity_forecast.monitor reference
# The daily check (writes live/monitoring.json):
#     uv run python -m se3_electricity_forecast.monitor
# Only pandas and NumPy are used here, no model library.

import json
import sys
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from se3_electricity_forecast.evaluate import TEST_END, TEST_START, TZ
from se3_electricity_forecast.features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "live"
RECORD = LIVE / "forecasts.csv"
INPUTS = LIVE / "inputs.csv"
STATUS = LIVE / "monitoring.json"
REFERENCE = ROOT / "results" / "monitoring_reference.json"
FORECAST_FEATURES = ROOT / "data" / "processed" / "features_forecast_weather.parquet"
LSTM_PREDICTIONS = ROOT / "data" / "processed" / "test_predictions_lstm.parquet"

ERROR_DAYS = 7          # the error check uses the last 7 counted days
ERROR_QUANTILE = 0.95   # alert when the error is worse than 95% of all 7-day periods in the test year
DRIFT_DAYS = 14         # the drift check uses the inputs of the last 14 forecast days
MIN_DRIFT_DAYS = 7      # and needs at least 7 of them
PSI_LIMIT = 0.25        # a population stability index above 0.25 is a common sign of real drift

# The calendar inputs follow the clock, so only prices and weather can drift
DRIFT_COLS = [c for c in FEATURE_COLS if c not in ("hour", "weekday", "month", "is_weekend", "is_holiday")]

TITLES = {
    "deadline": "Monitoring alert: deadline missed",
    "error": "Monitoring alert: live error above limit",
    "drift": "Monitoring alert: input drift",
}


# Reads a CSV file from the live record with its times as timestamps. round_trip reads every
# number back exactly as it was written, so saving the file again does not change it.
def read_live_csv(path):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, float_precision="round_trip")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["time_local"] = df["time_utc"].dt.tz_convert(TZ)
    df["issued_at_utc"] = pd.to_datetime(df["issued_at_utc"], utc=True)
    return df


# Sum of absolute errors and number of hours per day (Swedish date)
def daily_errors(time_local, actual, forecast):
    table = pd.DataFrame({"day": time_local.dt.date.to_numpy(), "error": (actual - forecast).abs().to_numpy(), "hours": 1})
    return table.groupby("day")[["error", "hours"]].sum()


# MAE over every window of `days` consecutive days, weighted by hours (23-, 24- and 25-hour days)
def rolling_mae(per_day, days):
    sums = per_day.rolling(days).sum()
    return (sums["error"] / sums["hours"]).dropna()


# Splits values into the bins of the reference and returns the share of values in each bin
def bin_shares(values, edges):
    counts = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1)
    return counts / counts.sum()


# Population stability index: how much a distribution has moved away from its reference.
# 0 means the same; above 0.25 is usually treated as a real shift.
def psi(values, edges, reference_shares):
    live = np.clip(bin_shares(values, edges), 1e-4, None)
    ref = np.clip(np.asarray(reference_shares), 1e-4, None)
    return float(np.sum((live - ref) * np.log(live / ref)))


# Builds the fixed limits from the test year: the error limit from the LSTM's test predictions,
# and for every month and input the bin edges (deciles) and the share of hours in each bin.
# The inputs are the ones with archived weather forecasts, like the live inputs.
def make_reference(features, predictions):
    preds = predictions.copy()
    preds["time_local"] = preds["time_utc"].dt.tz_convert(TZ)
    per_day = daily_errors(preds["time_local"], preds["actual"], preds["prediction"])
    windows = rolling_mae(per_day, ERROR_DAYS)

    test = features[(features["time_local"] >= TEST_START) & (features["time_local"] < TEST_END)]
    months = {}
    for month, rows in test.groupby(test["time_local"].dt.month):
        months[str(month)] = {}
        for col in DRIFT_COLS:
            values = rows[col].to_numpy(float)
            edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
            months[str(month)][col] = {"edges": edges.tolist(), "shares": bin_shares(values, edges).tolist()}
    return {
        "error_limit": float(windows.quantile(ERROR_QUANTILE)),
        "error_median": float(windows.median()),
        "psi_limit": PSI_LIMIT,
        "test_period": [str(TEST_START.date()), str((TEST_END - pd.Timedelta(days=1)).date())],
        "months": months,
    }


def check_deadline(record, now):
    today = now.tz_convert(TZ).date()
    day = today + timedelta(days=1)
    rows = record[record["time_local"].dt.date == day] if record is not None else pd.DataFrame()
    if len(rows) and rows["issued_before_noon"].all():
        made = rows["issued_at_utc"].min().tz_convert(TZ)
        return {"status": "ok", "day": str(day), "message": f"The forecast for {day} was made at {made:%H:%M}, before 12:00."}
    if now >= pd.Timestamp(today, tz=TZ) + pd.Timedelta(hours=12):
        return {"status": "alert", "day": str(day), "message": f"No forecast for {day} was made before 12:00 Swedish time."}
    return {"status": "waiting", "day": str(day), "message": f"The forecast for {day} has not been made yet. It is still before 12:00."}


def check_error(record, limit):
    if record is None:
        return {"status": "waiting", "days": 0, "message": "There is no live record yet."}
    done = record[record["actual"].notna() & record["issued_before_noon"]]
    lstm = daily_errors(done["time_local"], done["actual"], done["lstm"])
    if len(lstm) < ERROR_DAYS:
        return {"status": "waiting", "days": len(lstm),
                "message": f"{len(lstm)} of {ERROR_DAYS} counted days so far. The check starts after {ERROR_DAYS} days."}
    naive = daily_errors(done["time_local"], done["actual"], done["weekly_naive"])
    last, last_naive = lstm.tail(ERROR_DAYS), naive.tail(ERROR_DAYS)
    mae = float(last["error"].sum() / last["hours"].sum())
    naive_mae = float(last_naive["error"].sum() / last_naive["hours"].sum())
    status = "alert" if mae > limit else "ok"
    word = "above" if status == "alert" else "within"
    return {"status": status, "days": len(lstm), "lstm_mae": mae, "weekly_naive_mae": naive_mae, "limit": limit,
            "message": f"LSTM MAE of the last {ERROR_DAYS} counted days: {mae:.3f} SEK/kWh, {word} the limit of "
                       f"{limit:.3f} (baseline: {naive_mae:.3f})."}


def check_drift(inputs, reference):
    days = sorted(inputs["time_local"].dt.date.unique()) if inputs is not None else []
    if len(days) < MIN_DRIFT_DAYS:
        return {"status": "waiting", "days": len(days),
                "message": f"{len(days)} of {MIN_DRIFT_DAYS} days of inputs so far. The check starts after {MIN_DRIFT_DAYS} days."}
    recent_days = days[-DRIFT_DAYS:]
    recent = inputs[inputs["time_local"].dt.date.isin(recent_days)]
    month = str(recent_days[-1].month)
    values = {col: psi(recent[col].to_numpy(float), reference["months"][month][col]["edges"],
                       reference["months"][month][col]["shares"]) for col in DRIFT_COLS}
    drifted = [col for col, value in sorted(values.items(), key=lambda kv: -kv[1]) if value > reference["psi_limit"]]
    status = "alert" if drifted else "ok"
    message = (f"These inputs differ from the same month in the test year: {', '.join(drifted)}." if drifted
               else "All inputs look like the same month in the test year.")
    return {"status": status, "days": len(recent_days), "month": int(month), "limit": reference["psi_limit"],
            "psi": values, "drifted": drifted, "message": message}


# Runs the three checks and lists an alert (issue title and text) for every check that failed
def run_checks(record, inputs, reference, now):
    checks = {
        "deadline": check_deadline(record, now),
        "error": check_error(record, reference["error_limit"]),
        "drift": check_drift(inputs, reference),
    }
    alerts = [{"title": TITLES[name], "body": f"{check['message']}\n\nChecked at {now:%Y-%m-%d %H:%M} UTC."}
              for name, check in checks.items() if check["status"] == "alert"]
    return {"checked_at_utc": now.isoformat(), "checks": checks, "alerts": alerts}


def main():
    if sys.argv[1:] == ["reference"]:
        reference = make_reference(pd.read_parquet(FORECAST_FEATURES), pd.read_parquet(LSTM_PREDICTIONS))
        REFERENCE.write_text(json.dumps(reference, indent=1) + "\n")
        print(f"saved {REFERENCE.relative_to(ROOT)}: 7-day error limit {reference['error_limit']:.4f} "
              f"(median {reference['error_median']:.4f}), {len(reference['months'])} months of input bins")
        return
    status = run_checks(read_live_csv(RECORD), read_live_csv(INPUTS), json.loads(REFERENCE.read_text()),
                        pd.Timestamp.now(tz="UTC"))
    STATUS.write_text(json.dumps(status, indent=2) + "\n")
    for name, check in status["checks"].items():
        print(f"{name:8s} {check['status']:7s} {check['message']}")


if __name__ == "__main__":
    main()