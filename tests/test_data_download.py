# Tests for data_download.py: turning 15-minute prices into hourly prices. No internet is used.

import pandas as pd
from se3_electricity_forecast.data_download import to_hourly


def test_to_hourly_averages_15_minute_prices_and_keeps_hourly_prices():
    raw = pd.DataFrame({
        # one old-style hourly price, then four 15-minute prices in the next hour
        "time_start": pd.to_datetime(["2025-09-30T23:00:00+02:00", "2025-10-01T00:00:00+02:00",
                                      "2025-10-01T00:15:00+02:00", "2025-10-01T00:30:00+02:00",
                                      "2025-10-01T00:45:00+02:00"], utc=True),
        "SEK_per_kWh": [5.0, 1.0, 2.0, 3.0, 4.0],
        "EUR_per_kWh": [0.5, 0.1, 0.2, 0.3, 0.4],
    })
    hourly = to_hourly(raw)
    assert list(hourly["price_sek_kwh"]) == [5.0, 2.5]
    assert list(hourly.columns) == ["time_utc", "price_sek_kwh", "price_eur_kwh", "time_local"]
    assert str(hourly["time_local"].dt.tz) == "Europe/Stockholm"