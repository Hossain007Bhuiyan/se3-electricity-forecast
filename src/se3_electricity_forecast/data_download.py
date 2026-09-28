"""Download SE3 electricity prices and Stockholm weather data.

Prices: elprisetjustnu.se (free, no API key). Data before 1 Oct 2025 is hourly,
after that it is 15-minute intervals, so we also build an hourly version.
Weather: Open-Meteo historical archive (free, no API key).
"""

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------- Settings ----------
ZONE = "SE3"
START_DATE = date(2022, 11, 1)  # First day with data in the price API

# The price API has one JSON file per day and zone, e.g. .../2026/09-26_SE3.json
PRICE_URL = (
    "https://www.elprisetjustnu.se/api/v1/prices/"
    "{year}/{month:02d}-{day:02d}_{zone}.json"
)
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

STOCKHOLM_LAT = 59.33
STOCKHOLM_LON = 18.07
WEATHER_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "shortwave_radiation",
]

# Find the project folder automatically, so the script works from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


# Downloads all prices for one day. Returns an empty list if the day is not available.
def fetch_prices_for_day(session: requests.Session, day: date) -> list[dict]:
    """Download all prices for one day. Returns an empty list if missing."""
    url = PRICE_URL.format(year=day.year, month=day.month, day=day.day, zone=ZONE)
    response = session.get(url, timeout=30)
    if response.status_code == 404:
        return []  # Day not available (e.g. tomorrow before 13:00)
    response.raise_for_status()
    return response.json()


# Downloads prices for every day from start to end (both included).
# Returns the prices as a table and a list of days that had no data.
def download_prices(start: date, end: date) -> tuple[pd.DataFrame, list[date]]:
    """Download prices for every day from start to end (inclusive)."""
    rows = []
    missing_days = []

    # A session reuses the same connection for all requests, which is faster
    with requests.Session() as session:
        day = start
        while day <= end:
            # If the internet hiccups, wait a little and try the same day once more
            try:
                data = fetch_prices_for_day(session, day)
            except requests.RequestException as error:
                print(f"  Problem on {day}: {error}. Trying again in 5 seconds...")
                time.sleep(5)
                data = fetch_prices_for_day(session, day)

            if data:
                rows.extend(data)
            else:
                missing_days.append(day)

            # Show progress once per month
            if day.day == 1:
                print(f"  Progress: reached {day}")

            day += timedelta(days=1)
            time.sleep(0.1)  # Be polite to the free API

    df = pd.DataFrame(rows)
    # utc=True is needed because Swedish times switch between +01:00 and +02:00
    df["time_start"] = pd.to_datetime(df["time_start"], utc=True)
    df["time_end"] = pd.to_datetime(df["time_end"], utc=True)
    df = (
        df.sort_values("time_start")
        .drop_duplicates(subset="time_start")
        .reset_index(drop=True)
    )
    return df, missing_days

# Since 1 Oct 2025 prices come in 15-minute intervals. This averages them into hourly
# prices, so the whole dataset has the same resolution. Older hourly data stays the same.
def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Average 15-minute prices into hourly prices (hourly data stays the same)."""
    hourly = (
        df.assign(time_utc=df["time_start"].dt.floor("h"))  ## floor("h") maps 12:00, 12:15, 12:30 and 12:45 to the same hour 12:00
        .groupby("time_utc", as_index=False)[["SEK_per_kWh", "EUR_per_kWh"]]
        .mean()
        .rename(columns={"SEK_per_kWh": "price_sek_kwh", "EUR_per_kWh": "price_eur_kwh"})
    )
    hourly["time_local"] = hourly["time_utc"].dt.tz_convert("Europe/Stockholm")  ## Keep Swedish local time as well, since daily habits follow the local clock
    return hourly


# Downloads hourly historical weather for Stockholm in a single request.
def download_weather(start: date, end: date) -> pd.DataFrame:
    """Download hourly historical weather for Stockholm in one request."""
    params = {
        "latitude": STOCKHOLM_LAT,
        "longitude": STOCKHOLM_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "UTC",
    }
    response = requests.get(WEATHER_URL, params=params, timeout=120)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df["time_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    # The newest few days may still be empty in the archive, so remove them
    df = df.dropna(subset=WEATHER_VARIABLES, how="all")
    return df[["time_utc", *WEATHER_VARIABLES]]


# ---------- Main ----------
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    tomorrow = today + timedelta(days=1)

  # Try up to tomorrow, since tomorrow's prices exist after about 13:00
    print(f"1/2 Downloading {ZONE} prices from {START_DATE} to {tomorrow}")
    print("    This takes about 3-6 minutes...")
    prices_raw, missing_days = download_prices(START_DATE, tomorrow)
    prices_hourly = to_hourly(prices_raw)
        # Save both versions: the raw one for checks, the hourly one for modelling
    prices_raw.to_parquet(RAW_DIR / "prices_se3_raw.parquet", index=False)
    prices_hourly.to_parquet(RAW_DIR / "prices_se3_hourly.parquet", index=False)
    print(f"    Saved {len(prices_raw):,} raw rows and {len(prices_hourly):,} hourly rows")
    if missing_days:
        print(f"    Days with no data: {[str(d) for d in missing_days]}")

    # The weather archive is a bit behind, so only ask for data up to yesterday
    print("2/2 Downloading Stockholm weather")
    weather = download_weather(START_DATE, today - timedelta(days=1))
    weather.to_parquet(RAW_DIR / "weather_stockholm_hourly.parquet", index=False)
    print(f"    Saved {len(weather):,} hourly weather rows")
    print(f"    Weather available until {weather['time_utc'].max()}")

    print("Done! Files are in data/raw/")


if __name__ == "__main__":
    main()