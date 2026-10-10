# Tests for monitor.py, the daily monitoring checks. Only synthetic data is used.

from datetime import date

import numpy as np
import pandas as pd
import pytest

from se3_electricity_forecast import monitor
from se3_electricity_forecast.evaluate import TEST_END, TEST_START

TZ = "Europe/Stockholm"


def hours(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start if start.tzinfo else start.tz_localize(TZ)
    end = end if end.tzinfo else end.tz_localize(TZ)
    time_utc = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    return time_utc, time_utc.tz_convert(TZ)


# Test-year inputs and LSTM predictions with random but realistic values
@pytest.fixture
def reference():
    rng = np.random.default_rng(1)
    time_utc, time_local = hours(TEST_START, TEST_END)
    features = pd.DataFrame({"time_utc": time_utc, "time_local": time_local})
    for col in monitor.DRIFT_COLS:
        features[col] = rng.normal(10, 2, len(time_utc))
    actual = rng.normal(1, 0.3, len(time_utc))
    predictions = pd.DataFrame({"time_utc": time_utc, "actual": actual, "prediction": actual + rng.normal(0, 0.2, len(time_utc))})
    return monitor.make_reference(features, predictions)


# A live record of `days` forecast days starting 1 Oct 2026, every forecast made at 06:05 the day before
def make_record(days, error=0.1, before_noon=True):
    time_utc, time_local = hours("2026-10-01", pd.Timestamp("2026-10-01") + pd.Timedelta(days=days))
    actual = np.full(len(time_utc), 1.0)
    issued = (time_local.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=6, minutes=5)).tz_convert("UTC")
    return pd.DataFrame({"time_utc": time_utc, "time_local": time_local, "issued_at_utc": issued,
                         "issued_before_noon": before_noon, "lstm": actual + error,
                         "weekly_naive": actual + 0.3, "actual": actual})


def make_inputs(days, shift=0.0):
    rng = np.random.default_rng(2)
    time_utc, time_local = hours("2026-10-01", pd.Timestamp("2026-10-01") + pd.Timedelta(days=days))
    inputs = pd.DataFrame({"time_utc": time_utc, "time_local": time_local})
    for col in monitor.DRIFT_COLS:
        inputs[col] = rng.normal(10 + shift, 2, len(time_utc))
    return inputs


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 5000)
    edges = np.quantile(values, np.linspace(0.1, 0.9, 9))
    shares = monitor.bin_shares(values, edges)
    assert monitor.psi(values, edges, shares) < 1e-9
    assert monitor.psi(values + 2, edges, shares) > 1


def test_reference_error_limit_and_month_bins(reference):
    assert sorted(int(m) for m in reference["months"]) == list(range(1, 13))
    assert reference["error_median"] < reference["error_limit"]
    bins = reference["months"]["10"]["temperature_2m"]
    assert len(bins["shares"]) == len(bins["edges"]) + 1 and np.isclose(sum(bins["shares"]), 1)


def test_deadline_check():
    record = make_record(3)  # forecasts for 1, 2 and 3 October
    ok = monitor.check_deadline(record, pd.Timestamp("2026-10-02 10:30", tz="UTC"))  # tomorrow = 3 Oct
    assert ok["status"] == "ok" and "06:05" in ok["message"]
    late = monitor.check_deadline(record, pd.Timestamp("2026-10-03 10:30", tz="UTC"))  # 12:30, no forecast for 4 Oct
    assert late["status"] == "alert"
    early = monitor.check_deadline(record, pd.Timestamp("2026-10-03 07:00", tz="UTC"))  # 09:00, still before 12:00
    assert early["status"] == "waiting"
    after_noon = make_record(3, before_noon=False)  # made, but too late to count
    assert monitor.check_deadline(after_noon, pd.Timestamp("2026-10-02 10:30", tz="UTC"))["status"] == "alert"


def test_error_check_uses_only_counted_days(reference):
    assert monitor.check_error(make_record(5), reference["error_limit"])["status"] == "waiting"
    ok = monitor.check_error(make_record(10, error=0.01), reference["error_limit"])
    assert ok["status"] == "ok" and np.isclose(ok["lstm_mae"], 0.01) and np.isclose(ok["weekly_naive_mae"], 0.3)
    assert monitor.check_error(make_record(10, error=5.0), reference["error_limit"])["status"] == "alert"
    late = make_record(10, error=5.0, before_noon=False)  # forecasts after 12:00 never count
    assert monitor.check_error(late, reference["error_limit"])["status"] == "waiting"


def test_drift_check(reference):
    assert monitor.check_drift(make_inputs(5), reference)["status"] == "waiting"
    same = monitor.check_drift(make_inputs(14), reference)
    assert same["status"] == "ok" and same["month"] == 10 and set(same["psi"]) == set(monitor.DRIFT_COLS)
    shifted = monitor.check_drift(make_inputs(14, shift=6), reference)
    assert shifted["status"] == "alert" and set(shifted["drifted"]) == set(monitor.DRIFT_COLS)


def test_every_failed_check_becomes_one_alert(reference):
    status = monitor.run_checks(make_record(10, error=5.0), make_inputs(14, shift=6), reference,
                                pd.Timestamp("2026-10-11 10:30", tz="UTC"))  # no forecast for 12 Oct
    assert [a["title"] for a in status["alerts"]] == list(monitor.TITLES.values())
    assert date.fromisoformat(status["checks"]["deadline"]["day"]) == date(2026, 10, 12)


def test_saving_the_record_again_does_not_change_it(tmp_path):
    path = tmp_path / "forecasts.csv"
    record = make_record(2)
    record["lstm"] = record["lstm"] + 0.123456789012345678  # numbers with many digits
    record.to_csv(path, index=False)
    first = path.read_text()
    monitor.read_live_csv(path).to_csv(path, index=False)
    assert path.read_text() == first