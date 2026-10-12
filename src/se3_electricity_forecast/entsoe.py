# Talks to the ENTSO-E Transparency Platform, where Europe's grid operators publish load
# forecasts, power production and cross-border flows. The API needs a personal token:
# on the Mac it comes from the .env file and on GitHub from the ENTSOE_TOKEN secret.

import os
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import requests

URL = "https://web-api.tp.entsoe.eu/api"
# The code ENTSO-E uses for the SE3 bidding zone (Stockholm)
SE3 = "10Y1001A1001A46L"
TZ = "Europe/Stockholm"


def get_token():
    token = os.environ.get("ENTSOE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("ENTSOE_TOKEN is not set. On the Mac, start the command with: uv run --env-file .env")
    return token


# Sends one request and returns the answer as XML text. The token is part of the web address,
# so error messages are written here without it. That way it never shows up in a terminal or a log.
def fetch(params):
    token = get_token()
    try:
        response = requests.get(URL, params={"securityToken": token, **params}, timeout=60)
    except requests.RequestException as error:
        # "from None" also hides the original error, because its text contains the full address
        raise RuntimeError(f"Could not reach ENTSO-E ({type(error).__name__}).") from None
    if response.status_code != 200:
        text = response.text.replace(token, "***")[:500]
        raise RuntimeError(f"ENTSO-E answered with HTTP {response.status_code}: {text}")
    return response.text


# ENTSO-E wants times in UTC, written as yyyyMMddHHmm
def utc_stamp(ts):
    return ts.tz_convert("UTC").strftime("%Y%m%d%H%M")


# Start and end of one Swedish day in ENTSO-E's format. DateOffset keeps the local midnight,
# so the days when the clocks change get their 23 or 25 hours.
def day_period(day):
    return utc_stamp(day), utc_stamp(day + pd.DateOffset(days=1))


# Counts the values in an answer and lists their time step (PT60M = hourly, PT15M = every 15 minutes)
def summarize(xml_text):
    root = ET.fromstring(xml_text)
    points = root.findall(".//{*}Point")
    steps = sorted({el.text for el in root.findall(".//{*}resolution")})
    return len(points), steps


# When there is no data, ENTSO-E sends a short note with the reason instead
def reason(xml_text):
    text = ET.fromstring(xml_text).find(".//{*}Reason/{*}text")
    return text.text if text is not None else ""

# Turns an answer into a table with one row per time step: the kind of series (the production
# type for generation, empty for other data), the start time in UTC, the value and the step length.
# With curve type A03 ENTSO-E leaves out values that repeat the one before, so those are filled in.
# Other gaps stay empty, so missing data is never hidden.
def parse(xml_text):
    parts = []
    for series in ET.fromstring(xml_text).findall(".//{*}TimeSeries"):
        psr = series.find(".//{*}psrType")
        kind = psr.text if psr is not None else ""
        # For generation, a series with an "out" zone is power used, e.g. for pumping water up
        if kind and series.find("{*}outBiddingZone_Domain.mRID") is not None:
            kind += "_consumption"
        curve = series.find("{*}curveType")
        for period in series.findall("{*}Period"):
            start = pd.Timestamp(period.find("{*}timeInterval/{*}start").text)
            end = pd.Timestamp(period.find("{*}timeInterval/{*}end").text)
            step = pd.Timedelta(period.find("{*}resolution").text)
            times = pd.date_range(start, end, freq=step, inclusive="left")
            values = pd.Series(float("nan"), index=range(1, len(times) + 1))
            for point in period.findall("{*}Point"):
                values[int(point.find("{*}position").text)] = float(point.find("{*}quantity").text)
            if curve is not None and curve.text == "A03":
                values = values.ffill()
            parts.append(pd.DataFrame({"kind": kind, "time_utc": times, "value": values.to_numpy(), "step": step}))
    if not parts:
        return pd.DataFrame(columns=["kind", "time_utc", "value", "step"])
    return pd.concat(parts, ignore_index=True)

# A quick test of the token: asks for today's day-ahead load forecast for SE3
def check():
    day = pd.Timestamp.now(tz=TZ).normalize()
    start, end = day_period(day)
    xml_text = fetch({
        "documentType": "A65",  # total load
        "processType": "A01",   # day-ahead forecast
        "outBiddingZone_Domain": SE3,
        "periodStart": start,
        "periodEnd": end,
    })
    count, steps = summarize(xml_text)
    if count == 0:
        print(f"The token works. ENTSO-E sent no values for {day:%Y-%m-%d}: {reason(xml_text)}")
        return
    print(f"The token works: {count} values of the SE3 load forecast for {day:%Y-%m-%d} (time step {', '.join(steps)}).")


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
    else:
        print("Usage: uv run --env-file .env python -m se3_electricity_forecast.entsoe check")