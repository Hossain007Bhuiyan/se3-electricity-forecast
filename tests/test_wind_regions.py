# Tests for wind_regions.py. Open-Meteo is faked, so no internet is used.

import numpy as np
import pandas as pd
from se3_electricity_forecast import wind_regions
from se3_electricity_forecast.wind_regions import COLUMNS


class FakeResponse:
    def __init__(self, data):
        self.status_code = 200
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def test_download_regions_asks_once_per_zone(monkeypatch):
    calls = []

    # Answers with the latitude as the wind speed, so every column shows which point it came from
    def fake_get(url, params, timeout):
        calls.append(params)
        return FakeResponse({"hourly": {"time": ["2026-10-01T00:00", "2026-10-01T01:00"],
                                        params["hourly"]: [params["latitude"], None]}})

    monkeypatch.setattr(wind_regions.requests, "get", fake_get)
    table = wind_regions.download_regions("https://example.com", "2026-10-01", "2026-10-01", "wind_speed_100m")
    assert list(table.columns) == ["time_utc", *COLUMNS]
    assert table["time_utc"].iloc[0] == pd.Timestamp("2026-10-01 00:00", tz="UTC")
    assert table["wind_SE1"].iloc[0] == 65.4 and table["wind_SE4"].iloc[0] == 55.9
    assert table["wind_SE2"].isna().iloc[1]  # a missing value stays empty
    assert len(calls) == 4 and all(call["hourly"] == "wind_speed_100m" for call in calls)


def test_drop_empty_end_keeps_empty_hours_in_the_middle():
    df = pd.DataFrame({col: [1.0, np.nan, 3.0, np.nan, np.nan] for col in COLUMNS})
    kept = wind_regions.drop_empty_end(df)
    assert len(kept) == 3  # the 2 empty hours at the end are removed
    assert kept["wind_SE1"].isna().sum() == 1  # the empty hour in the middle stays and counts as missing


def test_fill_small_gaps_fills_at_most_3_hours():
    values = [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan, 11.0]
    df = pd.DataFrame({col: values for col in COLUMNS})
    filled, missing = wind_regions.fill_small_gaps(df.copy())
    assert missing == 7 * len(COLUMNS)  # all missing values are counted and reported
    assert filled["wind_SE2"].iloc[1:3].tolist() == [2.0, 3.0]  # a 2-hour gap is filled
    assert filled["wind_SE2"].isna().sum() == 2  # only 3 hours of the 5-hour gap are filled


def test_compare_shows_error_and_bias():
    times = pd.date_range("2026-10-01", periods=3, freq="h", tz="UTC")
    actual = pd.DataFrame({"time_utc": times, **{col: [10.0, 20.0, 30.0] for col in COLUMNS}})
    forecast = pd.DataFrame({"time_utc": times, **{col: [12.0, 18.0, 34.0] for col in COLUMNS}})
    result = wind_regions.compare(actual, forecast).set_index("zone")
    assert np.isclose(result.loc["wind_SE3", "mae"], 8 / 3)
    assert np.isclose(result.loc["wind_SE3", "bias"], 4 / 3)