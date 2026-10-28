# Tests for entsoe.py. No real requests are sent: the answers from ENTSO-E are faked,
# so the tests need no token and run anywhere, also on GitHub.

import pandas as pd
import pytest
import requests
from se3_electricity_forecast import entsoe

# A shortened answer in ENTSO-E's format with two hourly values
LOAD = """<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <Period>
      <resolution>PT60M</resolution>
      <Point><position>1</position><quantity>9000</quantity></Point>
      <Point><position>2</position><quantity>9100</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>"""

# The note ENTSO-E sends when there is no data
NO_DATA = """<?xml version="1.0" encoding="UTF-8"?>
<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0">
  <Reason><code>999</code><text>No matching data found</text></Reason>
</Acknowledgement_MarketDocument>"""


class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def test_missing_token_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("ENTSOE_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ENTSOE_TOKEN is not set"):
        entsoe.get_token()


def test_http_error_never_shows_the_token(monkeypatch):
    monkeypatch.setenv("ENTSOE_TOKEN", "secret-123")
    monkeypatch.setattr(entsoe.requests, "get", lambda *args, **kwargs: FakeResponse(401, "Invalid token secret-123"))
    with pytest.raises(RuntimeError) as error:
        entsoe.fetch({"documentType": "A65"})
    assert "401" in str(error.value)
    assert "secret-123" not in str(error.value)


def test_connection_error_never_shows_the_token(monkeypatch):
    monkeypatch.setenv("ENTSOE_TOKEN", "secret-123")

    # A real connection error from requests contains the full address with the token
    def fail(*args, **kwargs):
        raise requests.ConnectionError("Max retries exceeded with url: /api?securityToken=secret-123")

    monkeypatch.setattr(entsoe.requests, "get", fail)
    with pytest.raises(RuntimeError) as error:
        entsoe.fetch({"documentType": "A65"})
    assert "secret-123" not in str(error.value)
    assert error.value.__suppress_context__  # the original error is not shown either


def test_summarize_and_reason():
    assert entsoe.summarize(LOAD) == (2, ["PT60M"])
    assert entsoe.summarize(NO_DATA) == (0, [])
    assert entsoe.reason(NO_DATA) == "No matching data found"


def test_day_period_follows_swedish_time():
    # 25 Oct 2026: summer time ends, so the Swedish day has 25 hours
    assert entsoe.day_period(pd.Timestamp("2026-10-25", tz="Europe/Stockholm")) == ("202610242200", "202610252300")
    # A normal winter day: midnight in Sweden is 23:00 UTC
    assert entsoe.day_period(pd.Timestamp("2026-11-10", tz="Europe/Stockholm")) == ("202611092300", "202611102300")


# Builds a small answer in ENTSO-E's format. A value of None leaves that point out.
def make_xml(start, step, values, curve="A01", psr=None):
    end = pd.Timestamp(start) + len(values) * pd.Timedelta(step)
    points = "".join(f"<Point><position>{i}</position><quantity>{v}</quantity></Point>"
                     for i, v in enumerate(values, start=1) if v is not None)
    psr_xml = f"<MktPSRType><psrType>{psr}</psrType></MktPSRType>" if psr else ""
    return (f'<GL_MarketDocument xmlns="urn:test"><TimeSeries><curveType>{curve}</curveType>{psr_xml}'
            f'<Period><timeInterval><start>{start}</start><end>{end:%Y-%m-%dT%H:%MZ}</end></timeInterval>'
            f'<resolution>{step}</resolution>{points}</Period></TimeSeries></GL_MarketDocument>')


def test_parse_fills_left_out_values_only_for_curve_a03():
    values = [5, None, 7, None]
    a03 = entsoe.parse(make_xml("2026-10-11T22:00Z", "PT15M", values, curve="A03"))
    assert a03["value"].tolist() == [5, 5, 7, 7]  # left-out values repeat the one before
    assert a03["time_utc"].iloc[1] == pd.Timestamp("2026-10-11 22:15", tz="UTC")
    a01 = entsoe.parse(make_xml("2026-10-11T22:00Z", "PT15M", values, curve="A01"))
    assert a01["value"].isna().sum() == 2  # real gaps stay empty


def test_parse_names_the_production_type():
    data = entsoe.parse(make_xml("2026-10-11T22:00Z", "PT60M", [100, 110], psr="B14"))
    assert data["kind"].unique().tolist() == ["B14"]
    assert entsoe.parse(NO_DATA).empty


def test_parse_counts_a_week_across_the_clock_change_once():
    # The week 19-25 Oct 2026 ends when summer time ends, so it is 7 days and 1 hour long
    week = ('<GL_MarketDocument xmlns="urn:test"><TimeSeries><curveType>A03</curveType><Period><timeInterval>'
            '<start>2026-10-18T22:00Z</start><end>2026-10-25T23:00Z</end></timeInterval><resolution>P7D</resolution>'
            '<Point><position>1</position><quantity>25000000</quantity></Point></Period></TimeSeries></GL_MarketDocument>')
    data = entsoe.parse(week)
    assert len(data) == 1 and data["time_utc"].iloc[0] == pd.Timestamp("2026-10-18 22:00", tz="UTC")
    # A day with 25 hours still gets all its 100 quarter-hours
    day = make_xml("2026-10-24T22:00Z", "PT15M", [1.0] * 100)
    assert len(entsoe.parse(day)) == 100