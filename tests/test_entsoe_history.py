# Tests for entsoe_history.py. ENTSO-E is faked, so the tests need no token.

import pandas as pd
import pytest
from se3_electricity_forecast import entsoe, entsoe_history
from test_entsoe import make_xml


def test_chunks_cover_the_period_without_gaps():
    start = pd.Timestamp("2022-10-01", tz="Europe/Stockholm")
    end = pd.Timestamp("2024-01-15", tz="Europe/Stockholm")
    parts = entsoe_history.chunks(start, end)
    assert len(parts) == 3
    assert parts[0] == (start, pd.Timestamp("2023-04-01", tz="Europe/Stockholm"))
    assert parts[-1][1] == end
    # each part starts where the one before ends
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))


def test_download_skips_parts_without_data_and_drops_doubles(monkeypatch):
    answers = [
        RuntimeError("ENTSO-E answered with HTTP 400: No matching data found"),
        make_xml("2023-03-31T22:00Z", "PT60M", [1, 2, 3]),
        make_xml("2023-04-01T00:00Z", "PT60M", [3, 4]),  # overlaps the part before by one hour
    ]

    def fake_fetch(params):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(entsoe, "fetch", fake_fetch)
    monkeypatch.setattr(entsoe_history.time, "sleep", lambda seconds: None)
    start = pd.Timestamp("2022-10-01", tz="Europe/Stockholm")
    data = entsoe_history.download({"documentType": "A65"}, start, start + pd.DateOffset(months=18))
    assert data["value"].tolist() == [1, 2, 3, 4]
    assert data["time_utc"].is_unique


def test_download_tries_once_more_and_then_stops(monkeypatch):
    calls = []

    def fail(params):
        calls.append(params)
        raise RuntimeError("Could not reach ENTSO-E (ConnectionError).")

    monkeypatch.setattr(entsoe, "fetch", fail)
    monkeypatch.setattr(entsoe_history.time, "sleep", lambda seconds: None)
    start = pd.Timestamp("2022-10-01", tz="Europe/Stockholm")
    with pytest.raises(RuntimeError, match="Could not reach"):
        entsoe_history.download({"documentType": "A65"}, start, start + pd.DateOffset(months=1))
    assert len(calls) == 2


def test_to_hourly_averages_quarters_per_series():
    quarters = make_xml("2026-10-11T22:00Z", "PT15M", [1, 2, 3, 4, None, None, None, None], psr="B19")
    nuclear = make_xml("2026-10-11T22:00Z", "PT60M", [5000, 5100], psr="B14")
    data = pd.concat([entsoe.parse(quarters), entsoe.parse(nuclear)])
    hourly = entsoe_history.to_hourly(data, "generation")
    assert list(hourly.columns) == ["generation_B14", "generation_B19"]
    assert hourly["generation_B19"].tolist()[0] == 2.5
    assert hourly.loc[pd.Timestamp("2026-10-11 23:00", tz="UTC"), "generation_B14"] == 5100
    assert pd.isna(hourly.loc[pd.Timestamp("2026-10-11 23:00", tz="UTC"), "generation_B19"])  # a fully missing hour stays empty