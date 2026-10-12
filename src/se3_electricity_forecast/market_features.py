# Adds the ENTSO-E market data and the wind in Sweden's four price zones to the feature table,
# for the challenger model. The forecast for a day is made at about 11:30 on the day before
# (the run day), so every value is moved in time to use only what exists at that moment:
# - the SE3 load forecast for the forecast day itself (published on the run day)
# - production per type and net import of the day before the run day (complete by then)
# - the newest weekly hydro reservoir level that is already published
# - wind at 100 m: measured in training, the forecast made 2 days ahead in validation and test
# The availability check shows at the end of October which of these inputs are kept.
# Run with: uv run python -m se3_electricity_forecast.market_features

from datetime import timedelta
from pathlib import Path
import pandas as pd
from se3_electricity_forecast.availability import NEIGHBOURS
from se3_electricity_forecast.evaluate import TZ, split
from se3_electricity_forecast.wind_regions import COLUMNS as WIND_COLS

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Production types used: B14 nuclear, B12 hydro, B19 wind, B16 solar (all in MW)
PRODUCTION = {"generation_B14": "nuclear", "generation_B12": "hydro", "generation_B19": "wind_power",
              "generation_B16": "solar"}
# Days from the end of a reservoir week (Monday 00:00) until the first run day that may use it.
# Set on the safe side for now. The availability check will show the real delay.
RESERVOIR_DELAY_DAYS = 4
MARKET_COLS = ["load_forecast", *[f"{name}_2d_ago" for name in PRODUCTION.values()],
               "net_import_2d_ago", "reservoir_twh", *WIND_COLS]


# Power flowing into SE3 minus power flowing out, summed over all 5 borders (MW).
# Sometimes one direction is missing while the other has a value. Most likely nothing
# flowed that way, so it counts as 0. If both directions are missing, the hour stays empty.
def net_import(entsoe):
    total = 0
    for zone in NEIGHBOURS:
        into, out = entsoe[f"flow_{zone}_to_SE3"], entsoe[f"flow_SE3_to_{zone}"]
        total = total + (into.fillna(0) - out.fillna(0)).mask(into.isna() & out.isna())
    return total


# Counts the hours where exactly one direction of a border is missing and how many of them
# have power flowing the other way. If almost all do, counting the missing one as 0 is right.
def check_missing_directions(entsoe):
    one_missing = other_flowing = 0
    for zone in NEIGHBOURS:
        into, out = entsoe[f"flow_{zone}_to_SE3"], entsoe[f"flow_SE3_to_{zone}"]
        for missing, other in [(into, out), (out, into)]:
            hours = missing.isna() & other.notna()
            one_missing += int(hours.sum())
            other_flowing += int((other[hours] > 0).sum())
    return one_missing, other_flowing


# Daily averages of production and net import. On the run day only the day before is complete,
# so each forecast day gets the values from two days before it. Example: the forecast for
# Thursday is made on Wednesday and uses Tuesday.
def two_days_before(entsoe):
    df = entsoe.assign(net_import=net_import(entsoe))
    names = {**PRODUCTION, "net_import": "net_import"}
    day = df["time_utc"].dt.tz_convert(TZ).dt.date
    daily = df.groupby(day)[list(names)].mean()
    daily.columns = [f"{name}_2d_ago" for name in names.values()]
    daily.index = [d + timedelta(days=2) for d in daily.index]
    return daily


# The newest reservoir level (TWh) each forecast day may use. A week ends on Monday 00:00 and
# may be used from RESERVOIR_DELAY_DAYS days later on the run day, so one day later for the forecast day.
def reservoir_by_day(reservoir, days):
    # Both date columns get the same type, otherwise merge_asof refuses to compare them
    weeks = pd.DataFrame({
        # +12 hours gives the Monday whether ENTSO-E starts the week at 22:00 or 23:00 UTC
        "usable_from": (pd.to_datetime((reservoir["week_start_utc"] + pd.Timedelta(hours=12)).dt.tz_convert(TZ).dt.date)
                        + pd.Timedelta(days=7 + RESERVOIR_DELAY_DAYS + 1)).astype("datetime64[ns]"),
        "reservoir_twh": reservoir["reservoir_SE"] / 1e6,  # MWh to TWh
    }).sort_values("usable_from")
    target = pd.DataFrame({"day": pd.to_datetime(sorted(set(days))).astype("datetime64[ns]")})
    merged = pd.merge_asof(target, weeks, left_on="day", right_on="usable_from")
    return merged.set_index(merged["day"].dt.date)["reservoir_twh"]


# Adds all market columns to a feature table. Missing values are not filled and no rows are
# dropped: LightGBM can handle empty values. Every gap stays visible.
def add_market_features(features, entsoe, reservoir, wind):
    df = features.merge(entsoe[["time_utc", "load_forecast"]], on="time_utc", how="left")
    day = df["time_local"].dt.date
    daily = two_days_before(entsoe)
    for col in daily.columns:
        df[col] = day.map(daily[col])
    df["reservoir_twh"] = day.map(reservoir_by_day(reservoir, day))
    return df.merge(wind[["time_utc", *WIND_COLS]], on="time_utc", how="left")


def main():
    entsoe = pd.read_parquet(RAW / "entsoe_se3_hourly.parquet")
    reservoir = pd.read_parquet(RAW / "entsoe_reservoir_se_weekly.parquet")
    # Training uses measured weather, validation and test use weather forecasts, as before
    tables = {
        "features_market.parquet": (pd.read_parquet(PROCESSED / "features.parquet"),
                                    pd.read_parquet(RAW / "wind_regions_hourly.parquet")),
        "features_market_forecast_weather.parquet": (pd.read_parquet(PROCESSED / "features_forecast_weather.parquet"),
                                                     pd.read_parquet(RAW / "wind_regions_forecast_hourly.parquet")),
    }
    for name, (features, wind) in tables.items():
        market = add_market_features(features, entsoe, reservoir, wind)
        market.to_parquet(PROCESSED / name, index=False)
        print(f"{name}: {len(market):,} rows, {len(features):,} before")
        missing = pd.DataFrame({part: data[MARKET_COLS].isna().sum()
                                for part, data in zip(["train", "valid", "test"], split(market))})
        print("Missing values per input:")
        print(missing.to_string())
        print()

    one_missing, other_flowing = check_missing_directions(entsoe)
    print(f"Hours with one flow direction missing: {one_missing}, with power flowing the other way: {other_flowing}")


if __name__ == "__main__":
    main()