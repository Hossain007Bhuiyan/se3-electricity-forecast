# Downloads the history of the ENTSO-E series that the availability check watches, for the
# whole period of the price data. The challenger model will learn from this history. It will
# only use the inputs the availability check shows are there before 12:00.
# Run on the Mac with: uv run --env-file .env python -m se3_electricity_forecast.entsoe_history

import time
from pathlib import Path
import pandas as pd
from se3_electricity_forecast import entsoe
from se3_electricity_forecast.availability import NEIGHBOURS, SWEDEN

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TZ = entsoe.TZ
# One month before the first price (1 Nov 2022), so even the first days have the values
# of the day and the week before
START = pd.Timestamp("2022-10-01", tz=TZ)

# The same requests as in the availability check, without the time window
SERIES = {
    "load_forecast": {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": entsoe.SE3},
    "generation": {"documentType": "A75", "processType": "A16", "in_Domain": entsoe.SE3},
}
for zone, code in NEIGHBOURS.items():
    SERIES[f"flow_{zone}_to_SE3"] = {"documentType": "A11", "in_Domain": entsoe.SE3, "out_Domain": code}
    SERIES[f"flow_SE3_to_{zone}"] = {"documentType": "A11", "in_Domain": code, "out_Domain": entsoe.SE3}
RESERVOIR = {"documentType": "A72", "processType": "A16", "in_Domain": SWEDEN}


# Splits the period into parts of at most six months, because ENTSO-E limits how much
# one request may ask for
def chunks(start, end):
    parts = []
    while start < end:
        stop = min(start + pd.DateOffset(months=6), end)
        parts.append((start, stop))
        start = stop
    return parts


# Downloads one series part by part. A part without data is skipped, e.g. before a series
# was first published. Any other error is tried once more after a short wait.
def download(params, start, end):
    parts = []
    for part_start, part_end in chunks(start, end):
        request = {**params, "periodStart": entsoe.utc_stamp(part_start), "periodEnd": entsoe.utc_stamp(part_end)}
        for attempt in range(2):
            try:
                parts.append(entsoe.parse(entsoe.fetch(request)))
                break
            except RuntimeError as error:
                if "No matching data" in str(error):
                    break
                if attempt == 1:
                    raise
                print(f"    {error} Trying again in 10 seconds...")
                time.sleep(10)
        time.sleep(0.5)  # be polite to the free API
    data = pd.concat(parts, ignore_index=True) if parts else entsoe.parse("<empty/>")
    # Parts can overlap at their edges, so every time step is kept only once
    return data.drop_duplicates(subset=["kind", "time_utc"], keep="last")


# Averages 15-minute values into hours, like the prices. Each series gets its own column,
# e.g. generation_B14. An hour where every value is missing stays empty.
def to_hourly(data, name):
    data = data.assign(series=[f"{name}_{kind}" if kind else name for kind in data["kind"]],
                       time_utc=data["time_utc"].dt.floor("h"))
    return data.pivot_table(index="time_utc", columns="series", values="value", aggfunc="mean")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    # Only full days: everything before today's midnight in Swedish time
    end = pd.Timestamp.now(tz=TZ).normalize()

    tables = []
    for i, (name, params) in enumerate(SERIES.items(), start=1):
        print(f"{i}/{len(SERIES) + 1} Downloading {name}")
        tables.append(to_hourly(download(params, START, end), name))
    hourly = pd.concat(tables, axis=1, sort=True)
    # Every hour of the period gets a row, so missing hours are visible as empty values
    hours = pd.date_range(START.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
    hourly = hourly.reindex(hours).rename_axis("time_utc").reset_index()
    hourly.to_parquet(RAW / "entsoe_se3_hourly.parquet", index=False)

    print(f"{len(SERIES) + 1}/{len(SERIES) + 1} Downloading reservoir_SE (weekly)")
    reservoir = download(RESERVOIR, START, end)
    reservoir = reservoir.rename(columns={"time_utc": "week_start_utc", "value": "reservoir_SE"})
    reservoir[["week_start_utc", "reservoir_SE"]].to_parquet(RAW / "entsoe_reservoir_se_weekly.parquet", index=False)

    print(f"Saved {len(hourly):,} hours from {hourly['time_utc'].min()} to {hourly['time_utc'].max()}")
    print("Missing hours per series:")
    print(hourly.drop(columns="time_utc").isna().sum().to_string())
    print(f"Saved {len(reservoir)} weeks of reservoir levels, the newest starting {reservoir['week_start_utc'].max()}")


if __name__ == "__main__":
    main()