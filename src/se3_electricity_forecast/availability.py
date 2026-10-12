# Records which ENTSO-E data for the next forecast really exists before the 12:00 deadline.
# It runs every day at about 11:30 for two weeks. Only inputs that are reliably there in time
# go into the challenger model, so it never uses data it could not have had.

import sys
from pathlib import Path
import pandas as pd
from se3_electricity_forecast import entsoe

TZ = entsoe.TZ
# The code for Sweden as a whole: hydro reservoir levels are published per country
SWEDEN = "10YSE-1--------K"
# The zones that have a power line to SE3
NEIGHBOURS = {
    "SE2": "10Y1001A1001A45N",
    "SE4": "10Y1001A1001A47J",
    "NO1": "10YNO-1--------2",
    "DK1": "10YDK-1--------W",
    "FI": "10YFI-1--------U",
}


# Start and end of one Swedish day in UTC
def day_window(day):
    return day.tz_convert("UTC"), (day + pd.DateOffset(days=1)).tz_convert("UTC")


def make_row(series, start, end, found, expected, covered_until, note):
    return {"series": series, "window_start_utc": start, "window_end_utc": end, "found": found,
            "expected": expected, "complete": expected > 0 and found == expected,
            "covered_until_utc": covered_until, "note": note}


# Asks ENTSO-E for one data item and returns one row per series in the answer: how many time
# steps of the window from start to end are there. Errors are saved as a note instead of
# stopping the whole check.
def check(name, params, start, end, fetch_from=None):
    params = {**params, "periodStart": entsoe.utc_stamp(fetch_from or start), "periodEnd": entsoe.utc_stamp(end)}
    try:
        xml_text = entsoe.fetch(params)
        data = entsoe.parse(xml_text)
    except RuntimeError as error:
        return [make_row(name, start, end, 0, 0, None, str(error))]
    if data.empty:
        return [make_row(name, start, end, 0, 0, None, entsoe.reason(xml_text) or "no data")]
    rows = []
    for kind, part in data.groupby("kind", sort=True):
        step = part["step"].iloc[0]
        inside = part[(part["time_utc"] >= start) & (part["time_utc"] < end)].dropna(subset=["value"])
        # round() because a day can have 23 or 25 hours and a week one hour more or less
        expected = round((end - start) / step)
        covered = (inside["time_utc"] + step).max() if len(inside) else None
        rows.append(make_row(f"{name}_{kind}" if kind else name, start, end,
                             inside["time_utc"].nunique(), expected, covered, ""))
    return rows


# Runs all checks for the forecast that is made now (now is a time in UTC)
def check_all(now):
    today = now.tz_convert(TZ).normalize()
    tomorrow_start, tomorrow_end = day_window(today + pd.DateOffset(days=1))
    yesterday_start, yesterday_end = day_window(today - pd.DateOffset(days=1))
    # Last week from Monday to Monday in Swedish time
    monday = today - pd.DateOffset(days=today.weekday())
    week_start, week_end = (monday - pd.DateOffset(days=7)).tz_convert("UTC"), monday.tz_convert("UTC")

    rows = check("load_forecast", {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": entsoe.SE3},
                 tomorrow_start, tomorrow_end)
    rows += check("generation", {"documentType": "A75", "processType": "A16", "in_Domain": entsoe.SE3},
                  yesterday_start, yesterday_end)
    for zone, code in NEIGHBOURS.items():
        rows += check(f"flow_{zone}_to_SE3", {"documentType": "A11", "in_Domain": entsoe.SE3, "out_Domain": code},
                      yesterday_start, yesterday_end)
        rows += check(f"flow_SE3_to_{zone}", {"documentType": "A11", "in_Domain": code, "out_Domain": entsoe.SE3},
                      yesterday_start, yesterday_end)
    # Asks for the last four weeks, so the newest published week is visible even when it is late
    rows += check("reservoir_SE", {"documentType": "A72", "processType": "A16", "in_Domain": SWEDEN},
                  week_start, week_end, fetch_from=week_start - pd.Timedelta(days=21))

    result = pd.DataFrame(rows)
    result.insert(0, "checked_at_utc", now.tz_convert("UTC"))
    return result


if __name__ == "__main__":
    result = check_all(pd.Timestamp.now(tz="UTC"))
    table = result[["series", "found", "expected", "complete", "note"]].copy()
    table["note"] = table["note"].str.slice(0, 60)
    print(f"Checked at {result['checked_at_utc'].iloc[0]:%Y-%m-%d %H:%M} UTC")
    print(table.to_string(index=False))
    # With a file name, the result is also added to that file (used by the daily workflow)
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        result.to_csv(path, mode="a", header=not path.exists(), index=False)