# Downloads wind at 100 m (about the hub height of a wind turbine) for one point in each Swedish
# price zone. Most Swedish wind power is in the north, while the current model only knows the
# weather in Stockholm. Two versions, the same way as for the Stockholm weather:
# - measured wind (weather reanalysis) for the whole period, for training
# - archived forecasts made 2 days ahead for the validation and test years, so the model is
#   checked with the kind of wind data it really gets in the morning
# Run with: uv run python -m se3_electricity_forecast.wind_regions

from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLE = "wind_speed_100m"
# The forecast made 2 days before each hour, the same rule as for the Stockholm weather
LEAD = "previous_day2"
# One point in an area with many wind farms in each zone. A point is a simplification:
# it follows the wind in that area, not the output of every single wind farm.
REGIONS = {
    "SE1": (65.4, 20.8),  # Markbygden, west of Piteå
    "SE2": (63.0, 16.0),  # central Norrland, between Sollefteå and Ånge
    "SE3": (58.3, 12.9),  # Västergötland, between Vänern and Vättern
    "SE4": (55.9, 13.6),  # central Skåne
}
COLUMNS = [f"wind_{zone}" for zone in REGIONS]
# In UTC. The ENTSO-E history starts at midnight on 1 Oct 2022 in Swedish time, which is
# still 30 September in UTC. The forecasts cover the validation and test years.
START = "2022-09-30"
FORECAST_START = "2024-09-30"


# Downloads one hourly variable for one point and returns it as a series with UTC times
def download(url, lat, lon, start, end, variable):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": variable, "timezone": "UTC"}
    response = requests.get(url, params=params, timeout=120)
    # On an error, print Open-Meteo's own explanation before stopping
    if response.status_code != 200:
        print(response.text)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    return pd.Series(hourly[variable], index=pd.to_datetime(hourly["time"], utc=True), dtype=float)


# One column per zone, e.g. wind_SE2
def download_regions(url, start, end, variable):
    table = pd.DataFrame({f"wind_{zone}": download(url, lat, lon, start, end, variable)
                          for zone, (lat, lon) in REGIONS.items()})
    return table.rename_axis("time_utc").reset_index()


# Removes the empty hours at the end, e.g. the newest days the archive does not have yet.
# Empty hours in the middle are kept, so they are counted as missing and not hidden.
def drop_empty_end(df):
    has_data = df[COLUMNS].notna().any(axis=1)
    return df.loc[:has_data[has_data].index.max()]


# Fills gaps of up to 3 hours with a straight line between the neighbouring values.
# Also returns how many values were missing, so nothing is hidden.
def fill_small_gaps(df):
    missing = int(df[COLUMNS].isna().sum().sum())
    df[COLUMNS] = df[COLUMNS].interpolate(limit=3)
    return df, missing


# Shows how far the forecast is from the measured wind in each zone (km/h).
# mae = typical size of the forecast error, bias = whether it is too high (+) or too low (-).
def compare(actual, forecast):
    both = actual.merge(forecast, on="time_utc", suffixes=("_actual", "_forecast"))
    rows = []
    for col in COLUMNS:
        diff = both[f"{col}_forecast"] - both[f"{col}_actual"]
        rows.append({"zone": col, "mean_actual": both[f"{col}_actual"].mean(),
                     "mean_forecast": both[f"{col}_forecast"].mean(),
                     "mae": diff.abs().mean(), "bias": diff.mean()})
    return pd.DataFrame(rows)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    print("1/2 Downloading measured wind at 100 m")
    actual = download_regions(ARCHIVE_URL, START, yesterday, VARIABLE)
    actual = drop_empty_end(actual)
    actual, missing_actual = fill_small_gaps(actual)
    actual.to_parquet(RAW / "wind_regions_hourly.parquet", index=False)

    print("2/2 Downloading archived wind forecasts made 2 days ahead")
    forecast = download_regions(PREVIOUS_RUNS_URL, FORECAST_START, yesterday, f"{VARIABLE}_{LEAD}")
    forecast = drop_empty_end(forecast)
    forecast, missing_forecast = fill_small_gaps(forecast)
    forecast.to_parquet(RAW / "wind_regions_forecast_hourly.parquet", index=False)

    for name, df, missing in [("measured", actual, missing_actual), ("forecast", forecast, missing_forecast)]:
        print(f"{name}: {len(df):,} hours from {df['time_utc'].min()} to {df['time_utc'].max()}, "
              f"{missing} missing values found, {int(df[COLUMNS].isna().sum().sum())} still missing after filling")
    print()
    print(compare(actual, forecast).round(2).to_string(index=False))


if __name__ == "__main__":
    main()