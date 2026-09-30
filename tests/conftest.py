# Shared test data. The tests never use the real downloaded data: they build small, synthetic
# tables with the same columns, so they run anywhere in a few seconds, also on GitHub.

import numpy as np
import pandas as pd
import pytest

TZ = "Europe/Stockholm"
WEATHER_COLS = ["temperature_2m", "wind_speed_10m", "precipitation", "cloud_cover", "shortwave_radiation"]


# Hourly prices from 1 Oct to 5 Nov 2025 in the same format as prices_se3_hourly.parquet.
# The period includes 26 Oct 2025, when Sweden switched from summer to winter time (a 25-hour day).
# Every price is different (1.00, 1.01, 1.02, ...), so the tests can check exactly which hour
# a value came from.
def make_prices(start="2025-10-01", end="2025-11-05"):
    time_utc = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq="h",
                             inclusive="left").tz_convert("UTC")
    price = 1 + np.arange(len(time_utc)) * 0.01
    return pd.DataFrame({
        "time_utc": time_utc,
        "price_sek_kwh": price,
        "price_eur_kwh": price / 11,
        "time_local": time_utc.tz_convert(TZ),
    })


# Hourly weather for the same hours, in the same format as weather_stockholm_hourly.parquet
def make_weather(prices):
    rng = np.random.default_rng(0)
    weather = pd.DataFrame({"time_utc": prices["time_utc"]})
    for col in WEATHER_COLS:
        weather[col] = rng.random(len(prices))
    return weather


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def weather(prices):
    return make_weather(prices)