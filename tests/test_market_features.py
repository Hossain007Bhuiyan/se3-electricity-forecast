# Tests for market_features.py: every market value must only use information that exists at
# 11:30 on the day before the forecast day. Small made-up tables, no downloads.

import numpy as np
import pandas as pd
from se3_electricity_forecast import market_features
from se3_electricity_forecast.availability import NEIGHBOURS
from se3_electricity_forecast.market_features import MARKET_COLS, WIND_COLS

TZ = "Europe/Stockholm"


# ENTSO-E hours from 1 to 9 Oct 2026 (Swedish time). Every value is the day of the month,
# so each feature shows exactly which day it came from.
def make_entsoe():
    times = pd.date_range("2026-10-01", "2026-10-10", freq="h", tz=TZ, inclusive="left")
    day = times.day.to_numpy().astype(float)
    df = pd.DataFrame({"time_utc": times.tz_convert("UTC"), "load_forecast": day * 1000})
    for col in market_features.PRODUCTION:
        df[col] = day
    for zone in NEIGHBOURS:
        df[f"flow_{zone}_to_SE3"] = day
        df[f"flow_SE3_to_{zone}"] = 0.0
    return df


def make_features(start, end):
    times = pd.date_range(start, end, freq="h", tz=TZ, inclusive="left")
    return pd.DataFrame({"time_utc": times.tz_convert("UTC"), "time_local": times})


def test_net_import_counts_a_missing_direction_as_zero_only_if_the_other_has_a_value():
    entsoe = make_entsoe().head(3)
    entsoe.loc[0, "flow_SE3_to_SE2"] = np.nan  # one direction missing: counts as 0
    entsoe.loc[1, ["flow_SE2_to_SE3", "flow_SE3_to_SE2"]] = np.nan  # both missing: hour stays empty
    result = market_features.net_import(entsoe)
    assert result.iloc[0] == 5.0 and pd.isna(result.iloc[1]) and result.iloc[2] == 5.0
    assert market_features.check_missing_directions(entsoe) == (1, 1)


def test_daily_values_come_from_two_days_before():
    features = make_features("2026-10-08", "2026-10-09")  # forecast day Thursday 8 Oct
    reservoir = pd.DataFrame({"week_start_utc": pd.to_datetime(["2026-09-27 22:00"], utc=True), "reservoir_SE": [25e6]})
    wind = features[["time_utc"]].assign(**{col: 10.0 for col in WIND_COLS})
    df = market_features.add_market_features(features, make_entsoe(), reservoir, wind)
    assert (df["nuclear_2d_ago"] == 6).all()  # Tuesday 6 Oct, never Wednesday or Thursday
    assert (df["net_import_2d_ago"] == 6 * len(NEIGHBOURS)).all()
    assert (df["load_forecast"] == 8000).all()  # the load forecast is for the forecast day itself
    assert list(df.columns[-len(MARKET_COLS):]) == MARKET_COLS


def test_reservoir_is_used_only_after_the_delay():
    # Two weeks: 21-28 Sep and 28 Sep - 5 Oct. The second ends on Monday 5 Oct, so the run day
    # Friday 9 Oct (4 days later) is the first that may use it, for the forecast day 10 Oct.
    reservoir = pd.DataFrame({"week_start_utc": pd.to_datetime(["2026-09-20 22:00", "2026-09-27 22:00"], utc=True),
                              "reservoir_SE": [24e6, 25e6]})
    days = pd.Series(pd.to_datetime(["2026-10-09", "2026-10-10"]).date)
    result = market_features.reservoir_by_day(reservoir, days)
    assert result.tolist() == [24.0, 25.0]


def test_missing_market_data_stays_empty_and_no_rows_are_dropped():
    features = make_features("2026-10-08", "2026-10-09")
    entsoe = make_entsoe()
    entsoe.loc[entsoe["time_utc"].dt.tz_convert(TZ).dt.day == 8, "load_forecast"] = np.nan
    reservoir = pd.DataFrame({"week_start_utc": pd.to_datetime(["2026-09-27 22:00"], utc=True), "reservoir_SE": [25e6]})
    wind = features[["time_utc"]].assign(**{col: 10.0 for col in WIND_COLS})
    df = market_features.add_market_features(features, entsoe, reservoir, wind)
    assert len(df) == len(features)
    assert df["load_forecast"].isna().all()