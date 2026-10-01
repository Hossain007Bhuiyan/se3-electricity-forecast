# Tests for features.py: holidays, lag features, previous-day features and the full feature table.

from datetime import date
import numpy as np
import pandas as pd
from se3_electricity_forecast.features import (
    FEATURE_COLS, add_calendar_features, add_lag_features, add_previous_day_features,
    build_features, swedish_holidays,
)


def test_swedish_holidays_2025():
    # The 13 official holidays without Sundays, plus Midsommarafton, Christmas Eve and New Year's Eve
    expected = {
        date(2025, 1, 1), date(2025, 1, 6), date(2025, 4, 18), date(2025, 4, 20), date(2025, 4, 21),
        date(2025, 5, 1), date(2025, 5, 29), date(2025, 6, 6), date(2025, 6, 8), date(2025, 6, 20),
        date(2025, 6, 21), date(2025, 11, 1), date(2025, 12, 24), date(2025, 12, 25), date(2025, 12, 26),
        date(2025, 12, 31),
    }
    assert swedish_holidays(range(2025, 2026)) == expected
    # A normal Sunday must not count as a holiday
    assert date(2025, 9, 28) not in swedish_holidays(range(2025, 2026))


def test_lag_features_come_from_earlier_hours(prices):
    df = add_lag_features(prices.copy())
    # Row i must get the price of row i-24, i-48 and i-168 (the data has no missing hours)
    for i in [168, 500, len(df) - 1]:
        assert df["price_lag_24h"].iloc[i] == df["price_sek_kwh"].iloc[i - 24]
        assert df["price_lag_48h"].iloc[i] == df["price_sek_kwh"].iloc[i - 48]
        assert df["price_lag_168h"].iloc[i] == df["price_sek_kwh"].iloc[i - 168]


def test_no_lag_comes_from_the_forecast_day_itself(prices):
    # On 26 Oct 2025 (a 25-hour day), the last hour is only 24 hours after midnight of the same
    # day. Its 24-hour lag must still come from the day before, never from the forecast day.
    df = add_lag_features(prices.copy())
    price_to_row = {round(p, 2): i for i, p in enumerate(prices["price_sek_kwh"])}
    for col in ["price_lag_24h", "price_lag_48h", "price_lag_168h"]:
        for i, value in enumerate(df[col]):
            if pd.notna(value):
                source = price_to_row[round(value, 2)]
                assert prices["time_local"].iloc[source].date() < prices["time_local"].iloc[i].date()
    last_hour = df[df["time_local"] == pd.Timestamp("2025-10-26 23:00", tz="Europe/Stockholm")]
    previous_23 = prices[prices["time_local"] == pd.Timestamp("2025-10-25 23:00", tz="Europe/Stockholm")]
    assert last_hour["price_lag_24h"].iloc[0] == previous_23["price_sek_kwh"].iloc[0]


def test_previous_day_features_use_only_the_day_before(prices):
    df = add_previous_day_features(prices.copy())
    day = df[df["time_local"].dt.date == date(2025, 10, 15)]
    previous = prices[prices["time_local"].dt.date == date(2025, 10, 14)]["price_sek_kwh"]
    # isclose, because averages can differ in the last decimal depending on how they are summed
    assert np.isclose(day["prev_day_mean"], previous.mean()).all()
    assert (day["prev_day_min"] == previous.min()).all()
    assert (day["prev_day_max"] == previous.max()).all()


def test_no_leakage_from_the_forecast_day(prices):
    # Changing every price on 15 Oct must not change any feature of 15 Oct itself: the features
    # of a day may only use prices that are known the morning before, i.e. up to the day before.
    changed = prices.copy()
    on_day = changed["time_local"].dt.date == date(2025, 10, 15)
    changed.loc[on_day, "price_sek_kwh"] += 100

    def features_of_day(p):
        df = add_previous_day_features(add_lag_features(add_calendar_features(p.copy())))
        cols = [c for c in FEATURE_COLS if c in df.columns]
        return df.loc[df["time_local"].dt.date == date(2025, 10, 15), cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(features_of_day(prices), features_of_day(changed))


def test_build_features_table(prices, weather):
    features = build_features(prices, weather)
    assert "price_eur_kwh" not in features.columns  # same as the target, must never be an input
    assert set(FEATURE_COLS) <= set(features.columns)
    assert features.isna().sum().sum() == 0
    # The first 7 days are dropped, because they have no week-old prices yet
    assert features["time_local"].min() == pd.Timestamp("2025-10-08", tz="Europe/Stockholm")