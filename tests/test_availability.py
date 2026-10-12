# Tests for availability.py. ENTSO-E is faked: every request gets a small made-up answer,
# so the tests need no token.

import pandas as pd
from se3_electricity_forecast import availability, entsoe
from test_entsoe import make_xml

BORDER_ERROR = "ENTSO-E answered with HTTP 400: no such border"


# Answers like ENTSO-E would at 11:30: tomorrow's load forecast is complete (with left-out
# repeated values), yesterday's wind has a gap, one border has no data and last week's
# reservoir level is published
def fake_fetch(params):
    start = pd.Timestamp(params["periodStart"], tz="UTC")
    end = pd.Timestamp(params["periodEnd"], tz="UTC")
    quarters = int((end - start) / pd.Timedelta("15min"))
    begin = f"{start:%Y-%m-%dT%H:%MZ}"
    if params["documentType"] == "A65":
        return make_xml(begin, "PT15M", [None if i % 2 else 9000 for i in range(quarters)], curve="A03")
    if params["documentType"] == "A75":
        nuclear = make_xml(begin, "PT60M", [5000] * (quarters // 4), psr="B14")
        wind = make_xml(begin, "PT60M", [800] * (quarters // 4 - 3), psr="B19")
        return nuclear.replace("</GL_MarketDocument>", wind[wind.index("<TimeSeries>"):])
    if params["documentType"] == "A11":
        if params["out_Domain"] == availability.NEIGHBOURS["FI"]:
            raise RuntimeError(BORDER_ERROR)
        return make_xml(begin, "PT60M", [300] * (quarters // 4))
    # Weekly reservoir levels: four weeks, starting at a Monday midnight in Swedish time
    return make_xml(begin, "P7D", [1, 2, 3, 4])


def test_check_all(monkeypatch):
    monkeypatch.setattr(entsoe, "fetch", fake_fetch)
    # Saturday 24 Oct 2026, 11:30 in Sweden. Tomorrow is the 25-hour day when summer time ends.
    result = availability.check_all(pd.Timestamp("2026-10-24 09:30", tz="UTC")).set_index("series")
    assert result.loc["load_forecast", ["found", "expected", "complete"]].tolist() == [100, 100, True]
    assert result.loc["load_forecast", "window_start_utc"] == pd.Timestamp("2026-10-24 22:00", tz="UTC")
    assert result.loc["generation_B14", "complete"]
    assert result.loc["generation_B19", ["found", "expected", "complete"]].tolist() == [21, 24, False]
    assert result.loc["generation_B19", "covered_until_utc"] == pd.Timestamp("2026-10-23 19:00", tz="UTC")
    assert result.loc["flow_SE2_to_SE3", "complete"]
    assert not result.loc["flow_FI_to_SE3", "complete"] and result.loc["flow_FI_to_SE3", "note"] == BORDER_ERROR
    assert result.loc["reservoir_SE", ["found", "expected", "complete"]].tolist() == [1, 1, True]
    assert len(result) == 14  # load, 2 production types, 10 border directions and the reservoir


def test_results_are_added_to_the_file(monkeypatch, tmp_path):
    monkeypatch.setattr(entsoe, "fetch", fake_fetch)
    path = tmp_path / "availability.csv"
    for day in ["2026-10-13", "2026-10-14"]:
        result = availability.check_all(pd.Timestamp(f"{day} 09:30", tz="UTC"))
        result.to_csv(path, mode="a", header=not path.exists(), index=False)
    saved = pd.read_csv(path)
    assert len(saved) == 28 and saved["checked_at_utc"].nunique() == 2