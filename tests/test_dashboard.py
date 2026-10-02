# Tests for the dashboard: the calculations in dashboard/live_data.py, and a full run of
# dashboard/app.py with Streamlit's test tool. No internet is used.

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASHBOARD))
import live_data  # noqa: E402

TZ = "Europe/Stockholm"


# A small live record in the same format as forecasts.csv: 2 days, the first one made before
# 12:00 (counts), the second one after 12:00 (does not count)
def make_record():
    hours = pd.date_range(pd.Timestamp("2026-10-03", tz=TZ), pd.Timestamp("2026-10-05", tz=TZ), freq="h",
                          inclusive="left").tz_convert("UTC")
    actual = np.linspace(0.2, 1.0, len(hours))
    return pd.DataFrame({
        "time_utc": hours.astype(str),
        "time_local": hours.tz_convert(TZ).astype(str),
        "issued_at_utc": ["2026-10-02 05:30:00+00:00"] * 24 + ["2026-10-03 13:00:00+00:00"] * 24,
        "issued_before_noon": [True] * 24 + [False] * 24,
        "model_month": "2026-10",
        "lstm": actual + 0.1,
        "weekly_naive": actual - 0.3,
        "actual": actual,
    })


@pytest.fixture
def record(tmp_path):
    path = tmp_path / "forecasts.csv"
    make_record().to_csv(path, index=False)
    return live_data.load_record(path)


def test_load_record_reads_times_and_flags(record):
    assert str(record["time_local"].dt.tz) == TZ
    assert record["issued_before_noon"].tolist() == [True] * 24 + [False] * 24


def test_web_files_are_downloaded_with_requests(monkeypatch):
    # Files from the internet go through requests (with its own certificate list), not pandas
    class FakeResponse:
        text = "model,mae\nlstm,0.2\n"

        def raise_for_status(self):
            pass

    calls = []
    monkeypatch.setattr(live_data.requests, "get", lambda url, timeout: calls.append(url) or FakeResponse())
    table = live_data.load_test_results("https://example.com/test_results.csv")
    assert calls == ["https://example.com/test_results.csv"]
    assert table["mae"].tolist() == [0.2]
    

def test_latest_day_and_overview(record):
    day, rows = live_data.latest_day(record)
    assert day == date(2026, 10, 4) and len(rows) == 24
    overview = live_data.day_overview(rows)
    assert overview["peak_time"].hour == 23 and overview["low_time"].hour == 0  # prices rise all day
    assert np.isclose(overview["mean"], rows["lstm"].mean())


def test_daily_errors_only_count_forecasts_made_before_noon(record):
    errors = live_data.daily_errors(record)
    assert errors["day"].tolist() == [date(2026, 10, 3)]  # 4 Oct was forecast after 12:00
    assert errors["hours"].iloc[0] == 24
    assert np.isclose(errors["lstm_mae"].iloc[0], 0.1) and np.isclose(errors["weekly_naive_mae"].iloc[0], 0.3)


def test_hourly_prices_average_15_minute_prices():
    raw = pd.DataFrame({
        "time_start": ["2025-09-30T23:00:00+02:00", "2025-10-01T00:00:00+02:00", "2025-10-01T00:15:00+02:00",
                       "2025-10-01T00:30:00+02:00", "2025-10-01T00:45:00+02:00"],
        "SEK_per_kWh": [5.0, 1.0, 2.0, 3.0, 4.0],
    })
    hourly = live_data.hourly_prices(raw)
    assert hourly["price"].tolist() == [5.0, 2.5]
    assert hourly["time_local"].dt.hour.tolist() == [23, 0]


def test_price_landscape_handles_clock_changes():
    hours = pd.date_range(pd.Timestamp("2025-10-25", tz=TZ), pd.Timestamp("2025-10-28", tz=TZ), freq="h",
                          inclusive="left").tz_convert("UTC")
    prices = pd.DataFrame({"time_utc": hours, "price": np.arange(len(hours), dtype=float),
                           "time_local": hours.tz_convert(TZ)})
    table = live_data.price_landscape(prices)
    assert table.shape == (3, 24)
    # 26 Oct 2025 has 02:00 twice (rows 26 and 27 of the data); the landscape shows their average
    assert table.loc[date(2025, 10, 26), 2] == (26 + 27) / 2


def test_fetch_recent_prices_skips_missing_days(monkeypatch):
    class FakeResponse:
        def __init__(self, status, rows):
            self.status_code, self.rows = status, rows

        def json(self):
            return self.rows

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        if url.endswith("10-03_SE3.json"):  # "tomorrow", not published yet
            return FakeResponse(404, [])
        day = url.split("/")[-1][:5]  # e.g. "10-01"
        return FakeResponse(200, [{"time_start": f"2026-{day}T{h:02d}:00:00+02:00", "SEK_per_kWh": 1.0} for h in range(24)])

    monkeypatch.setattr(live_data.requests, "get", fake_get)
    prices = live_data.fetch_recent_prices(days=2, today=date(2026, 10, 2))
    assert len(prices) == 3 * 24  # 30 Sep, 1 Oct and 2 Oct; 3 Oct is skipped


def test_dashboard_page_runs(record, monkeypatch):
    # Replace every download with test data, then run the whole page like a browser would
    test_results = pd.DataFrame({"model": ["lstm", "lgbm_price_l1", "weekly_naive"], "hours": [8664] * 3,
                                 "mae": [0.21, 0.24, 0.32], "rmse": [0.30, 0.33, 0.44], "rel_mae": [0.66, 0.75, 1.0]})
    hours = pd.date_range(pd.Timestamp("2026-09-02", tz=TZ), pd.Timestamp("2026-10-03", tz=TZ), freq="h",
                          inclusive="left").tz_convert("UTC")
    recent = pd.DataFrame({"time_utc": hours, "price": np.linspace(0.1, 1.5, len(hours)), "time_local": hours.tz_convert(TZ)})
    monkeypatch.setattr(live_data, "load_record", lambda: record)
    monkeypatch.setattr(live_data, "load_test_results", lambda: test_results)
    monkeypatch.setattr(live_data, "fetch_recent_prices", lambda days: recent)

    page = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=60)
    page.run()
    assert not page.exception and not page.error
    # newest forecast, live accuracy, the chosen earlier day, the 3D landscape and the test results
    assert len(page.get("plotly_chart")) == 5
    assert "Saturday 03 October 2026" in page.selectbox[0].options  # the day with real prices can be chosen