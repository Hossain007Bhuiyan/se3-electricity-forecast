# Daily live forecast. Every morning, this forecasts tomorrow's hourly SE3 prices with the LSTM,
# and keeps a record of how earlier forecasts compared with the real prices.
#
# One run does four things:
#   1. update the price history (after the first run, only the newest days are downloaded)
#   2. retrain the LSTM once a month, the same way as in the walk-forward evaluation (Step 6B)
#   3. forecast every hour of tomorrow (Swedish time) with the LSTM and the weekly_naive baseline,
#      using a real weather forecast and only prices that are already published
#   4. add the forecast to the live record and fill in the real prices of earlier forecasts
#
# Run it with:  uv run python -m se3_electricity_forecast.daily
# Results go to the live/ folder. On GitHub, that folder is stored in a separate branch.
# Only PyTorch is loaded here, never LightGBM (see final_test.py for why).

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from se3_electricity_forecast import data_download, train_lstm
from se3_electricity_forecast.baselines import baseline_predictions
from se3_electricity_forecast.evaluate import TZ
from se3_electricity_forecast.features import (
    FEATURE_COLS, add_calendar_features, add_lag_features, add_previous_day_features, build_features,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "live"                       # downloaded data for the daily runs (not in Git)
LIVE = ROOT / "live"                                # results: model, forecasts and summary
PRICES = DATA / "prices_se3_hourly.parquet"
MODEL = LIVE / "model" / "lstm.pt"
RECORD = LIVE / "forecasts.csv"
SUMMARY = LIVE / "summary.json"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_COLS = data_download.WEATHER_VARIABLES



# A saved file and a fresh download can describe UTC with two different Python time zone objects.
# Combining them would turn the times into plain objects, and later merges would fail. This
# converts the times to one and the same UTC time zone, without changing any time.
def to_same_utc(prices):
    prices = prices.copy()
    prices["time_utc"] = prices["time_utc"].dt.tz_convert("UTC")
    prices["time_local"] = prices["time_utc"].dt.tz_convert(TZ)
    return prices


# Updates the local price history and returns it. The first time, everything since November 2022
# is downloaded. After that, only the last two known days and the newest days are downloaded again,
# which is much faster and kinder to the free price API.
def update_prices(end_day):
    DATA.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(PRICES) if PRICES.exists() else None
    start = data_download.START_DATE if old is None else old["time_local"].max().date() - timedelta(days=2)
    raw, _ = data_download.download_prices(start, end_day)
    new = to_same_utc(data_download.to_hourly(raw))
    if old is not None:
        # Keep the old hours before the re-downloaded part, then add the fresh download
        old = to_same_utc(old)
        new = pd.concat([old[old["time_utc"] < new["time_utc"].min()], new], ignore_index=True)
    prices = new.sort_values("time_utc").reset_index(drop=True)

    # The lag features need one row for every hour, so a gap must stop the run
    expected = pd.date_range(prices["time_utc"].min(), prices["time_utc"].max(), freq="h")
    if len(expected) != len(prices):
        raise ValueError(f"price history has {len(expected) - len(prices)} missing hours")
    prices.to_parquet(PRICES, index=False)
    return prices


# Downloads the hourly weather forecast for Stockholm for today and the next two days (in UTC),
# from Open-Meteo's normal forecast API, with the same variables and units as the training data.
def download_weather_forecast():
    params = {
        "latitude": data_download.STOCKHOLM_LAT,
        "longitude": data_download.STOCKHOLM_LON,
        "hourly": ",".join(WEATHER_COLS),
        "timezone": "UTC",
        "forecast_days": 3,
    }
    response = requests.get(FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"])
    df["time_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    return df[["time_utc", *WEATHER_COLS]]


# Builds the feature rows for every hour of `day` (Swedish date), exactly like build_features()
# does for the training data, but without knowing the prices of that day. Only prices from
# before the day are used, so this is what is really known in the morning before.
def day_features(prices, weather, day):
    start = pd.Timestamp(day, tz=TZ)
    end = pd.Timestamp(day + timedelta(days=1), tz=TZ)
    known = prices[prices["time_local"] < start]
    if known.empty or known["time_local"].max() != start - pd.Timedelta(hours=1):
        raise ValueError(f"the prices of the day before {day} are not complete yet")

    # Empty rows for the hours to forecast (23, 24 or 25 hours, depending on the clock change)
    hours = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    future = pd.DataFrame({"time_utc": hours, "price_sek_kwh": np.nan, "time_local": hours.tz_convert(TZ)})

    # The same steps as in build_features(), on the known prices plus the empty rows
    df = pd.concat([known.drop(columns=["price_eur_kwh"]), future], ignore_index=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_previous_day_features(df)
    df = df.merge(weather, on="time_utc", how="left")

    rows = df[df["time_local"] >= start].reset_index(drop=True)
    if rows[FEATURE_COLS].isna().any().any():
        raise ValueError(f"some inputs for {day} are missing (check the weather forecast)")
    return rows


# Trains the LSTM for the month that starts on month_start, exactly as in the walk-forward
# evaluation: all hours before (month_start - 60 days) for training, the 60 days after that for
# early stopping. Uses measured weather, like the original training.
def train_model(prices, weather, month_start):
    features = build_features(prices, weather)
    df = train_lstm.add_sequence_end(train_lstm.add_cyclical(features), prices)
    holdout_start = month_start - pd.Timedelta(days=train_lstm.HOLDOUT_DAYS)
    train = df[df["time_local"] < holdout_start]
    holdout = df[(df["time_local"] >= holdout_start) & (df["time_local"] < month_start)]
    model, stats, epochs = train_lstm.fit(train, holdout, prices["price_sek_kwh"].to_numpy())
    print(f"trained the LSTM for {month_start:%Y-%m}: {epochs} epochs, "
          f"data until {holdout['time_local'].max()}")
    return model, stats, epochs


# Saves the model and its scaling values. Only plain numbers, lists and text are stored next to
# the weights, so the file can be loaded safely with torch.load(weights_only=True).
def save_model(model, stats, month, epochs):
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    mu, sd, col_mean, col_std = stats
    torch.save({
        "state_dict": model.state_dict(),
        "mu": float(mu),
        "sd": float(sd),
        "col_mean": [float(v) for v in col_mean[train_lstm.INPUT_COLS]],
        "col_std": [float(v) for v in col_std[train_lstm.INPUT_COLS]],
        "input_cols": list(train_lstm.INPUT_COLS),
        "month": month,
        "epochs": epochs,
        "torch_version": str(torch.__version__),
    }, MODEL)


# Loads a saved model. Returns (model, stats, month), or None if there is no saved model.
def load_model():
    if not MODEL.exists():
        return None
    saved = torch.load(MODEL, weights_only=True)
    if saved["input_cols"] != list(train_lstm.INPUT_COLS):
        raise ValueError("the saved model was trained with different inputs")
    model = train_lstm.PriceLSTM(len(train_lstm.INPUT_COLS))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    # np.float64, the same type as during training, so predictions are calculated identically
    stats = (np.float64(saved["mu"]), np.float64(saved["sd"]),
             pd.Series(saved["col_mean"], index=train_lstm.INPUT_COLS),
             pd.Series(saved["col_std"], index=train_lstm.INPUT_COLS))
    return model, stats, saved["month"]


# Adds new forecasts to the record and fills in the real prices once they are known.
# A forecast for an hour that is already in the record is not replaced: the first forecast
# for each hour is the one that counts, so repeating a run cannot improve the record afterwards.
def update_record(record, forecast, prices):
    if record is not None:
        forecast = forecast[~forecast["time_utc"].isin(record["time_utc"])]
        record = pd.concat([record, forecast], ignore_index=True)
    else:
        record = forecast.copy()
    actual = prices.set_index("time_utc")["price_sek_kwh"]
    record["actual"] = record["time_utc"].map(actual)
    return record.sort_values("time_utc").reset_index(drop=True)


# Live accuracy so far, using only hours whose real price is known and whose forecast was made
# before 12:00 Swedish time on the day before (when bids for the day-ahead market close).
def summarize(record):
    done = record[record["actual"].notna() & record["issued_before_noon"]]
    summary = {"hours": int(len(done)), "days": int(done["time_local"].dt.date.nunique())}
    if len(done):
        lstm_mae = float((done["actual"] - done["lstm"]).abs().mean())
        naive_mae = float((done["actual"] - done["weekly_naive"]).abs().mean())
        summary.update({"lstm_mae": lstm_mae, "weekly_naive_mae": naive_mae, "rel_mae": lstm_mae / naive_mae})
    return summary


# Reads the record from CSV, with the time columns turned back into timestamps
def read_record():
    if not RECORD.exists():
        return None
    record = pd.read_csv(RECORD)
    record["time_utc"] = pd.to_datetime(record["time_utc"], utc=True)
    record["time_local"] = record["time_utc"].dt.tz_convert(TZ)
    record["issued_at_utc"] = pd.to_datetime(record["issued_at_utc"], utc=True)
    return record


# `now` can be given for testing; normally the current time is used
def main(now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    today = now.tz_convert(TZ).date()
    day = today + timedelta(days=1)  # the day to forecast
    print(f"forecasting {day} (Swedish time), run started {now:%Y-%m-%d %H:%M} UTC")

    prices = update_prices(day)
    print(f"price history: {len(prices)} hours, last hour {prices['time_local'].max()}")

    # Retrain at the start of every month, like in the walk-forward evaluation
    month_start = pd.Timestamp(date(day.year, day.month, 1), tz=TZ)
    month = f"{month_start:%Y-%m}"
    saved = load_model()
    if saved is None or saved[2] != month:
        weather_history = data_download.download_weather(data_download.START_DATE, today - timedelta(days=1))
        model, stats, epochs = train_model(prices, weather_history, month_start)
        save_model(model, stats, month, epochs)
    else:
        model, stats, _ = saved
        print(f"using the saved LSTM for {month}")

    # Inputs for tomorrow: published prices only, plus the newest weather forecast
    known = prices[prices["time_local"] < pd.Timestamp(day, tz=TZ)].reset_index(drop=True)
    rows = day_features(known, download_weather_forecast(), day)
    rows = train_lstm.add_sequence_end(train_lstm.add_cyclical(rows), known)

    forecast = pd.DataFrame({
        "time_utc": rows["time_utc"],
        "time_local": rows["time_local"],
        "issued_at_utc": now,
        "issued_before_noon": now < pd.Timestamp(today, tz=TZ) + pd.Timedelta(hours=12),
        "model_month": month,
        "lstm": train_lstm.predict_prices(model, stats, rows, known["price_sek_kwh"].to_numpy()),
        "weekly_naive": np.asarray(baseline_predictions(rows)["weekly_naive"]),
    })
    print()
    print(forecast[["time_local", "lstm", "weekly_naive"]].round({"lstm": 3, "weekly_naive": 3}).to_string(index=False))

    record = update_record(read_record(), forecast, prices)
    LIVE.mkdir(parents=True, exist_ok=True)
    record.to_csv(RECORD, index=False)
    summary = summarize(record)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print()
    print("live accuracy so far:", summary)


if __name__ == "__main__":
    main()