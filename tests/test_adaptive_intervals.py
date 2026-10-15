# Tests for adaptive_intervals.py. Only synthetic data is used.

import numpy as np
import pandas as pd
import pytest
from se3_electricity_forecast import adaptive_intervals as ai
from se3_electricity_forecast import intervals

TZ = "Europe/Stockholm"


# Forecasts for 1 Jan to 1 Jul 2025. From April the errors become three times larger, like a
# market that suddenly gets more volatile. Evening hours (18-24) always have larger errors.
@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    time_utc = pd.date_range(pd.Timestamp("2025-01-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ), freq="h",
                             inclusive="left").tz_convert("UTC")
    time_local = time_utc.tz_convert(TZ)
    size = np.where(time_local >= pd.Timestamp("2025-04-01", tz=TZ), 0.3, 0.1)
    size = size * np.where(time_local.hour >= 18, 3.0, 1.0)
    prediction = rng.uniform(0.5, 1.5, len(time_utc))
    preds = pd.DataFrame({"time_utc": time_utc, "prediction": prediction,
                          "actual": prediction + size * rng.normal(0, 1, len(time_utc))})
    # The current ranges read these two columns too, so they are added with a neutral value
    features = pd.DataFrame({"time_utc": time_utc, "time_local": time_local, "prev_day_std": 1.0, "prev_day_mean": 1.0})
    return preds, features


def coverage(df):
    return ((df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])).mean()


def test_adapting_brings_coverage_back_after_a_change(data):
    preds, features = data
    start, end = pd.Timestamp("2025-02-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ)
    fixed = ai.adaptive_intervals(preds, features, start, end, gamma=0.0, mondrian=False)
    adaptive = ai.adaptive_intervals(preds, features, start, end, gamma=0.05, mondrian=False)
    after = pd.Timestamp("2025-05-01", tz=TZ)  # one month after the errors grew
    assert coverage(fixed[fixed["time_local"] >= after]) < 0.7  # the fixed range stays too narrow
    assert abs(coverage(adaptive[adaptive["time_local"] >= after]) - 0.8) < 0.05


def test_mondrian_gives_the_evening_its_own_wider_range(data):
    preds, features = data
    df = ai.adaptive_intervals(preds, features, pd.Timestamp("2025-02-01", tz=TZ), pd.Timestamp("2025-03-01", tz=TZ),
                               gamma=0.0, mondrian=True)
    width = (df["upper"] - df["lower"]).groupby(df["time_local"].dt.hour >= 18).mean()
    assert width[True] > 2 * width[False]
    assert abs(coverage(df[df["time_local"].dt.hour >= 18]) - 0.8) < 0.06  # each block covers about 80%
    assert abs(coverage(df[df["time_local"].dt.hour < 18]) - 0.8) < 0.06


def test_a_day_only_uses_earlier_days(data):
    preds, features = data
    start, end = pd.Timestamp("2025-03-01", tz=TZ), pd.Timestamp("2025-03-15", tz=TZ)
    before = ai.adaptive_intervals(preds, features, start, end, gamma=0.05, mondrian=True)
    changed = preds.copy()
    later = changed["time_utc"] >= pd.Timestamp("2025-03-10", tz=TZ)
    changed.loc[later, "actual"] += 100  # huge errors from 10 March must not change the days before
    after = ai.adaptive_intervals(changed, features, start, end, gamma=0.05, mondrian=True)
    first = before["time_local"] < pd.Timestamp("2025-03-11", tz=TZ)  # 10 March itself only uses earlier days
    assert np.allclose(before.loc[first, "upper"], after.loc[first, "upper"])
    assert not np.allclose(before.loc[~first, "upper"], after.loc[~first, "upper"])


def test_days_without_enough_earlier_errors_are_skipped(data):
    preds, features = data
    df = ai.adaptive_intervals(preds, features, pd.Timestamp("2025-01-01", tz=TZ), pd.Timestamp("2025-03-01", tz=TZ),
                               gamma=0.03, mondrian=False)
    assert df["time_local"].min() == pd.Timestamp("2025-01-29", tz=TZ)  # the first day with 28 earlier days


def test_all_versions_use_the_same_hours_and_the_rule_picks_the_smallest_gap(data):
    preds, features = data
    result = ai.compare_versions(preds, features, pd.Timestamp("2025-02-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ))
    assert result["hours"].nunique() == 1 and len(result) == len(ai.VERSIONS)
    current = intervals.interval_metrics(intervals.add_intervals(preds, features, "constant",
                                         pd.Timestamp("2025-02-01", tz=TZ), pd.Timestamp("2025-07-01", tz=TZ)))
    assert np.isclose(result.set_index("version").loc["constant (current)", "coverage"], current["coverage"])
    chosen = ai.choose(result)
    assert chosen["month_gap"].iloc[0] == result["month_gap"].min()
    assert chosen["version"].iloc[0] != "constant (current)"


def test_monthly_coverage_on_a_tiny_example():
    df = pd.DataFrame({"time_local": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-02-01", "2025-02-02"]).tz_localize(TZ),
                       "actual": [1.0, 5.0, 1.0, 1.0], "lower": 0.0, "upper": 2.0})
    m = ai.monthly_coverage(df)
    assert np.isclose(m["month_gap"], (0.3 + 0.2) / 2)  # January 50%, February 100%
    assert np.isclose(m["worst_month"], 0.5)


# Saved test-year errors for the days before 1 Mar 2026 and one forecast day, 1 Mar 2026
def live_setup(tmp_path, monkeypatch, days=60):
    rng = np.random.default_rng(1)
    start = pd.Timestamp("2026-03-01", tz=TZ) - pd.Timedelta(days=days)
    times = pd.date_range(start, pd.Timestamp("2026-03-01", tz=TZ), freq="h", inclusive="left").tz_convert("UTC")
    path = tmp_path / "interval_scores.csv"
    pd.DataFrame({"time_utc": times, "score": rng.normal(0, 0.1, len(times)).round(6)}).to_csv(path, index=False)
    monkeypatch.setattr(ai, "SCORES", path)
    day = pd.date_range(pd.Timestamp("2026-03-01", tz=TZ), periods=24, freq="h")
    forecast = pd.DataFrame({"time_utc": day.tz_convert("UTC"), "time_local": day, "lstm": 1.0})
    return pd.read_csv(path)["score"], forecast


def test_live_range_on_the_first_run_uses_the_saved_errors(tmp_path, monkeypatch):
    scores, forecast = live_setup(tmp_path, monkeypatch)
    ranges = ai.live_ranges(None, forecast)
    assert np.allclose(ranges["lower"], 1.0 + np.quantile(scores, 0.1))
    assert np.allclose(ranges["upper"], 1.0 + np.quantile(scores, 0.9))


def test_live_range_learns_from_earlier_live_days_only(tmp_path, monkeypatch):
    _, forecast = live_setup(tmp_path, monkeypatch, days=90)
    first = ai.live_ranges(None, forecast)
    # Ten live days before the forecast day whose prices were far above every forecast
    days = pd.date_range(pd.Timestamp("2026-02-19", tz=TZ), pd.Timestamp("2026-03-01", tz=TZ), freq="h", inclusive="left")
    record = pd.DataFrame({"time_utc": days.tz_convert("UTC"), "time_local": days, "lstm": 1.0, "actual": 2.0})
    learned = ai.live_ranges(record, forecast)
    assert (learned["upper"] > first["upper"]).all()  # the upper edge moved up after the misses
    # Prices of the forecast day itself (known to a run after 13:00) must not change anything
    late = pd.concat([record, forecast.assign(actual=50.0)], ignore_index=True)
    assert np.allclose(ai.live_ranges(late, forecast)["upper"], learned["upper"])


def test_live_range_stays_empty_without_enough_errors(tmp_path, monkeypatch):
    _, forecast = live_setup(tmp_path, monkeypatch, days=10)
    ranges = ai.live_ranges(None, forecast)
    assert len(ranges) == 24 and ranges["lower"].isna().all()