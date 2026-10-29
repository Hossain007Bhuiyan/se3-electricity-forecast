# The challenger in live use, next to the LSTM (shadow mode). Every day at about 11:30,
# when the market data for tomorrow is published, it forecasts tomorrow with the challenger
# (LightGBM with market inputs, challenger_l2) and saves the average of the LSTM's forecast and
# the challenger's forecast. The LSTM stays the main forecast on the dashboard during the test.
#
# The model is trained once on all data up to the end of the test year and stays the same during
# the test, so retraining cannot change the result while it runs. The 18 inputs it shares with the
# LSTM are read from the daily run's saved inputs, so both models see exactly the same values.
# If a market input is missing (an ENTSO-E outage, a late publication), that day falls back to the
# LSTM's forecast, so a forecast is never missing. Such days are marked as fallback.
#     uv run python -m se3_electricity_forecast.challenger_live train            once, on the Mac
#     uv run --env-file .env python -m se3_electricity_forecast.challenger_live dry   test run, saves nothing
#     uv run python -m se3_electricity_forecast.challenger_live                  the daily run on GitHub
# No PyTorch here: the LSTM's forecast is read from the daily run's record.

import logging
import sys
from datetime import timedelta
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from se3_electricity_forecast import wind_regions
from se3_electricity_forecast.entsoe_history import RESERVOIR, SERIES, download, to_hourly
from se3_electricity_forecast.evaluate import TZ
from se3_electricity_forecast.features import FEATURE_COLS
from se3_electricity_forecast.market_features import MARKET_COLS, add_market_features
from se3_electricity_forecast.train_lgbm import PARAMS

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "features_market.parquet"
MODEL = ROOT / "models" / "challenger_l2.txt"
LIVE = ROOT / "live"
RECORD = LIVE / "forecasts.csv"
INPUTS = LIVE / "inputs.csv"
CHALLENGER = LIVE / "challenger.csv"
# The same files on GitHub, for the test run on the Mac
GITHUB_DATA = "https://raw.githubusercontent.com/Hossain007Bhuiyan/se3-electricity-forecast/forecast-data"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
COLS = FEATURE_COLS + MARKET_COLS
OBJECTIVE = "l2"  # challenger_l2 had the lowest MAE of the two challengers on the validation year

log = logging.getLogger(__name__)


# Trains the challenger once on every row of the feature table (measured weather and wind, like
# in the walk-forward evaluation) and saves it as a small text file
def train():
    df = pd.read_parquet(FEATURES)
    model = lgb.LGBMRegressor(objective=OBJECTIVE, **PARAMS).fit(df[COLS], df["price_sek_kwh"])
    MODEL.parent.mkdir(exist_ok=True)
    model.booster_.save_model(MODEL)
    return df


# Reads a live CSV file (a local path or a GitHub address) with its times as timestamps
def read_live(source):
    df = pd.read_csv(source, float_precision="round_trip")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["time_local"] = df["time_utc"].dt.tz_convert(TZ)
    df["issued_at_utc"] = pd.to_datetime(df["issued_at_utc"], utc=True)
    return df


# Downloads the market inputs for one forecast day, the same way as the history was downloaded:
# the load forecast for the day itself, production and flows of the day before the run day,
# the reservoir levels of the last weeks and the wind forecast for the four zones.
def market_inputs(day):
    start = pd.Timestamp(day, tz=TZ)
    end = start + pd.DateOffset(days=1)
    before = start - pd.DateOffset(days=2)
    tables = [to_hourly(download(SERIES["load_forecast"], start, end), "load_forecast")]
    for name, params in SERIES.items():
        if name != "load_forecast":
            tables.append(to_hourly(download(params, before, before + pd.DateOffset(days=1)), name))
    entsoe = pd.concat(tables, axis=1, sort=True).rename_axis("time_utc").reset_index()
    reservoir = download(RESERVOIR, start - pd.Timedelta(days=49), start)
    reservoir = reservoir.rename(columns={"time_utc": "week_start_utc", "value": "reservoir_SE"})
    wind = wind_regions.download_regions(FORECAST_URL, str(start.tz_convert("UTC").date()),
                                         str(end.tz_convert("UTC").date()), wind_regions.VARIABLE)
    return entsoe, reservoir, wind


# Tomorrow's forecast: the LSTM's forecast from the record, the challenger's forecast and their
# average. If anything is missing for the challenger, the average falls back to the LSTM.
def make_forecast(record, inputs, day, now, model, fetch=market_inputs):
    lstm = record.loc[record["time_local"].dt.date == day, ["time_utc", "time_local", "lstm"]]
    if lstm.empty:
        raise RuntimeError(f"the LSTM has no forecast for {day} yet")
    forecast = lstm.assign(issued_at_utc=now,
                           issued_before_noon=now < pd.Timestamp(now.tz_convert(TZ).date(), tz=TZ) + pd.Timedelta(hours=12),
                           challenger=np.nan, note="")
    try:
        rows = add_market_features(inputs[inputs["time_local"].dt.date == day], *fetch(day))
        missing = [col for col in COLS if rows[col].isna().any()]
        # Every hour of the LSTM's forecast needs its inputs, in the same order
        if missing or list(rows["time_utc"]) != list(lstm["time_utc"]):
            raise ValueError(f"missing inputs: {', '.join(missing) or 'some hours'}")
        forecast["challenger"] = model.predict(rows[COLS])
    except Exception as error:
        forecast["note"] = f"fallback to the LSTM: {type(error).__name__}: {error}"[:300]
    forecast["fallback"] = forecast["challenger"].isna()
    forecast["combined"] = np.where(forecast["fallback"], forecast["lstm"], (forecast["lstm"] + forecast["challenger"]) / 2)
    return forecast


# Like the daily record: the first forecast for each hour counts and is never replaced
def update_saved(saved, forecast):
    if saved is not None:
        forecast = pd.concat([saved, forecast[~forecast["time_utc"].isin(saved["time_utc"])]], ignore_index=True)
    return forecast.sort_values("time_utc").reset_index(drop=True)


def main():
    if sys.argv[1:] == ["train"]:
        df = train()
        log.info(f"trained on {len(df):,} hours from {df['time_local'].min()} to {df['time_local'].max()}, "
                 f"saved {MODEL.relative_to(ROOT)} ({MODEL.stat().st_size / 1e6:.1f} MB)")
        return
    dry = sys.argv[1:] == ["dry"]
    now = pd.Timestamp.now(tz="UTC")
    day = now.tz_convert(TZ).date() + timedelta(days=1)
    record = read_live(f"{GITHUB_DATA}/forecasts.csv" if dry else RECORD)
    inputs = read_live(f"{GITHUB_DATA}/inputs.csv" if dry else INPUTS)
    forecast = make_forecast(record, inputs, day, now, lgb.Booster(model_file=str(MODEL)))
    log.info(f"forecast for {day}:\n" + forecast[["time_local", "lstm", "challenger", "combined"]].round(
        {"lstm": 3, "challenger": 3, "combined": 3}).to_string(index=False))
    if forecast["fallback"].any():
        log.warning(forecast["note"].iloc[0])
    if dry:
        log.info("test run: nothing saved")
        return
    saved = read_live(CHALLENGER) if CHALLENGER.exists() else None
    update_saved(saved, forecast).to_csv(CHALLENGER, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()