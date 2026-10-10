# Tests for intervals.py, the 80% ranges around the LSTM forecasts. Only synthetic data is used.

import numpy as np
import pandas as pd
import pytest
from se3_electricity_forecast import intervals

TZ = "Europe/Stockholm"


# Forecasts for 1 Jan to 1 Jul 2025 whose errors grow with prev_day_std, like real price errors
# that are larger after volatile days
@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    time_utc = pd.date_range(pd.Timestamp("2025-01-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ), freq="h",
                             inclusive="left").tz_convert("UTC")
    days = len(time_utc) // 24 + 1
    daily_std = np.repeat(rng.uniform(0.05, 1.0, days), 24)[:len(time_utc)]
    features = pd.DataFrame({"time_utc": time_utc, "time_local": time_utc.tz_convert(TZ),
                             "prev_day_std": daily_std, "prev_day_mean": 1.0})
    prediction = rng.uniform(0.5, 1.5, len(time_utc))
    preds = pd.DataFrame({"time_utc": time_utc, "prediction": prediction,
                          "actual": prediction + daily_std * rng.normal(0, 1, len(time_utc))})
    return preds, features


def test_ranges_cover_about_80_percent(data):
    preds, features = data
    df = intervals.add_intervals(preds, features, "prev_day_std", pd.Timestamp("2025-02-01", tz=TZ),
                                 pd.Timestamp("2025-07-01", tz=TZ))
    m = intervals.interval_metrics(df)
    assert abs(m["coverage"] - 0.8) < 0.03 and (df["lower"] <= df["upper"]).all()


def test_the_right_scale_wins(data):
    preds, features = data
    result = intervals.compare_scales(preds, features, pd.Timestamp("2025-02-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ))
    assert result["scale"].iloc[0] == "prev_day_std"  # the errors really grow with prev_day_std
    assert set(result["scale"]) == set(intervals.SCALES)


def test_a_month_only_uses_earlier_errors(data):
    preds, features = data
    start, end = pd.Timestamp("2025-03-01", tz=TZ), pd.Timestamp("2025-04-01", tz=TZ)
    before = intervals.add_intervals(preds, features, "constant", start, end)
    changed = preds.copy()
    later = features.set_index("time_utc").loc[changed["time_utc"], "time_local"].to_numpy() >= start
    changed.loc[later, "actual"] += 100  # huge errors from March on must not change March's ranges
    after = intervals.add_intervals(changed, features, "constant", start, end)
    assert np.allclose(before["upper"] - before["prediction"], after["upper"] - after["prediction"])


def test_months_without_enough_earlier_errors_are_skipped(data):
    preds, features = data
    df = intervals.add_intervals(preds, features, "constant", pd.Timestamp("2025-01-01", tz=TZ), pd.Timestamp("2025-03-01", tz=TZ))
    assert df["time_local"].min() >= pd.Timestamp("2025-02-01", tz=TZ)  # January has no earlier errors


def test_metrics_on_a_tiny_example():
    df = pd.DataFrame({"actual": [1.0, 2.0, 3.0, 10.0], "lower": [0.5, 1.5, 3.5, 2.0], "upper": [1.5, 2.5, 4.5, 4.0]})
    m = intervals.interval_metrics(df)
    assert m["coverage"] == 0.5 and m["width"] == 1.25
    # interval score: width + 10 x miss below + 10 x miss above, averaged
    assert np.isclose(m["interval_score"], (1 + 1 + (1 + 10 * 0.5) + (2 + 10 * 6)) / 4)


def test_calibration_for_live_use(data):
    preds, features = data
    cal = intervals.calibration(preds, features, "prev_day_std", pd.Timestamp("2025-07-01", tz=TZ))
    assert cal["scale"] == "prev_day_std" and cal["q_low"] < 0 < cal["q_high"]
    assert cal["calibrated_on"] == ["2025-01-01", "2025-06-30"]