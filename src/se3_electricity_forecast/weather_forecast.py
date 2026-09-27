from pathlib import Path

import pandas as pd
import requests

from se3_electricity_forecast.evaluate import split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT = 59.33
LON = 18.07
START = "2024-09-30"
END = "2026-09-26"
LEAD = "previous_day2"
WEATHER_COLS = ["temperature_2m", "wind_speed_10m", "precipitation", "cloud_cover", "shortwave_radiation"]


def download_forecasts():
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START,
        "end_date": END,
        "hourly": ",".join(f"{col}_{LEAD}" for col in WEATHER_COLS),
        "timezone": "UTC",
    }
    response = requests.get(URL, params=params, timeout=120)
    if response.status_code != 200:
        print(response.text)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df = df.rename(columns={f"{col}_{LEAD}": col for col in WEATHER_COLS})
    return df[["time_utc", *WEATHER_COLS]]


def fill_small_gaps(df):
    missing = int(df[WEATHER_COLS].isna().sum().sum())
    df[WEATHER_COLS] = df[WEATHER_COLS].interpolate(limit=3)
    return df, missing


def use_forecast_weather(features, forecast):
    df = features.drop(columns=WEATHER_COLS).merge(forecast, on="time_utc", how="inner")
    return df.dropna().reset_index(drop=True)


def compare(actual, forecast):
    both = actual.merge(forecast, on="time_utc", suffixes=("_actual", "_forecast"))
    rows = []
    for col in WEATHER_COLS:
        diff = both[f"{col}_forecast"] - both[f"{col}_actual"]
        rows.append({
            "variable": col,
            "mean_actual": both[f"{col}_actual"].mean(),
            "mean_forecast": both[f"{col}_forecast"].mean(),
            "mae": diff.abs().mean(),
            "bias": diff.mean(),
        })
    return pd.DataFrame(rows)


def main():
    forecast = download_forecasts()
    forecast, missing = fill_small_gaps(forecast)
    forecast.to_parquet(RAW / "weather_forecast_stockholm_hourly.parquet", index=False)

    actual = pd.read_parquet(RAW / "weather_stockholm_hourly.parquet")
    features = pd.read_parquet(PROCESSED / "features.parquet")
    features_fc = use_forecast_weather(features, forecast)
    features_fc.to_parquet(PROCESSED / "features_forecast_weather.parquet", index=False)

    _, valid, test = split(features_fc)
    print("forecast hours:", len(forecast), "from", forecast["time_utc"].min(), "to", forecast["time_utc"].max())
    print("missing values filled:", missing)
    print("valid rows:", len(valid), "test rows:", len(test))
    print()
    print(compare(actual, forecast).round(3).to_string(index=False))


if __name__ == "__main__":
    main()