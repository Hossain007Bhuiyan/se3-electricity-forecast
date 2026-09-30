# Test data for the PyTorch tests. These tests live in their own folder and run in their own
# pytest process, because LightGBM and PyTorch must not run in the same Python process on macOS.
# The data is built the same way as in tests/conftest.py.

import numpy as np
import pandas as pd
import pytest
from se3_electricity_forecast.features import build_features

TZ = "Europe/Stockholm"
WEATHER_COLS = ["temperature_2m", "wind_speed_10m", "precipitation", "cloud_cover", "shortwave_radiation"]


# Hourly prices from 1 Oct to 5 Nov 2025, including the switch to winter time on 26 Oct.
# Every price is different (1.00, 1.01, 1.02, ...), so the tests can see where a value came from.
@pytest.fixture
def prices():
    time_utc = pd.date_range(pd.Timestamp("2025-10-01", tz=TZ), pd.Timestamp("2025-11-05", tz=TZ), freq="h",
                             inclusive="left").tz_convert("UTC")
    price = 1 + np.arange(len(time_utc)) * 0.01
    return pd.DataFrame({
        "time_utc": time_utc,
        "price_sek_kwh": price,
        "price_eur_kwh": price / 11,
        "time_local": time_utc.tz_convert(TZ),
    })


# The feature table built from those prices, with random weather
@pytest.fixture
def features(prices):
    rng = np.random.default_rng(0)
    weather = pd.DataFrame({"time_utc": prices["time_utc"]})
    for col in WEATHER_COLS:
        weather[col] = rng.random(len(prices))
    return build_features(prices, weather)