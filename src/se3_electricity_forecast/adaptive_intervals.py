# This steps adaptive 80% ranges. The current ranges (intervals.py) use one fixed pair of error
# quantiles for every hour. When prices behave differently than in the calibration year, the range
# is too narrow for weeks. Two known improvements are compared here:
#
# - Adaptive conformal prediction (Gibbs and Candes, 2021): after every day the quantile levels move
#   a little. If more than 10% of the prices fell below the range, the lower level goes down (wider
#   range below). If fewer did, it goes up again. The same for the upper side. gamma is the step size.
# - Separate ranges per time of day ("Mondrian" conformal prediction): four blocks of 6 hours
#   (00-06, 06-12, 12-18, 18-24), each with its own errors and its own levels.
#
# The range for a day only uses errors and misses from earlier days: on the morning a forecast is
# made, the prices of that day and all days before are already published. The version is chosen on
# the validation year with a rule fixed before running (see choose). The test year is not used again:
# a new version is judged on the live record, next to the current one.
#     uv run python -m se3_electricity_forecast.adaptive_intervals          compare on the validation year
#     uv run python -m se3_electricity_forecast.adaptive_intervals scores   save the test-year errors for live use
# Step 14D.2: the chosen version runs next to the current ranges in the daily forecast (live_ranges).
# Only pandas and NumPy are used here, no model library.

import sys
import numpy as np
import pandas as pd
from se3_electricity_forecast.evaluate import TEST_START, TZ, VALID_START
from se3_electricity_forecast.intervals import (CAL_DAYS, FEATURES, LOWER_Q, MIN_CAL_DAYS, RESULTS, TEST_PREDICTIONS,
                                                UPPER_Q, VALID_PREDICTIONS, add_intervals, interval_metrics)

TARGET = UPPER_Q - LOWER_Q  # 0.8
# gamma = 0 means no adapting, so "mondrian" alone is also in the list
VERSIONS = [{"name": "constant (current)"}]
for gamma in [0.01, 0.03, 0.05]:
    VERSIONS.append({"name": f"adaptive g={gamma}", "gamma": gamma, "mondrian": False})
VERSIONS.append({"name": "mondrian", "gamma": 0.0, "mondrian": True})
for gamma in [0.01, 0.03, 0.05]:
    VERSIONS.append({"name": f"adaptive + mondrian g={gamma}", "gamma": gamma, "mondrian": True})
# The winner on the validation year (results/intervals_adaptive_validation.csv)
CHOSEN = {"gamma": 0.05, "mondrian": False}
# The LSTM's errors on the test year. The daily run on GitHub has no data/ folder, so these are saved here.
SCORES = RESULTS / "interval_scores.csv"


def time_block(time_local):
    return time_local.dt.hour // 6


# Ranges for every day from start to end, built one day at a time like in live use.
# Days with fewer than MIN_CAL_DAYS earlier days of errors get no range.
def adaptive_intervals(preds, features, start, end, gamma, mondrian):
    df = preds.merge(features[["time_utc", "time_local"]], on="time_utc").sort_values("time_utc")
    df["score"] = df["actual"] - df["prediction"]
    df["day"] = df["time_local"].dt.normalize()
    df["group"] = time_block(df["time_local"]) if mondrian else 0
    levels = {}  # per group: [lower level, upper level], starting at 0.1 and 0.9
    parts = []
    for day in sorted(df.loc[(df["time_local"] >= start) & (df["time_local"] < end), "day"].unique()):
        earlier = df[(df["day"] < day) & (df["day"] >= day - pd.Timedelta(days=CAL_DAYS))]
        if earlier["day"].nunique() < MIN_CAL_DAYS:
            continue
        today = df[df["day"] == day].copy()
        for group, rows in today.groupby("group"):
            level = levels.setdefault(group, [LOWER_Q, UPPER_Q])
            scores = earlier.loc[earlier["group"] == group, "score"]
            # A level can move below 0 or above 1. Then the smallest or largest earlier error is used.
            q_low, q_high = np.quantile(scores, np.clip(level, 0, 1))
            lower = rows["prediction"] + q_low
            upper = rows["prediction"] + q_high
            today.loc[rows.index, "lower"] = lower
            today.loc[rows.index, "upper"] = upper
            # The day's misses are known before the next forecast is made, so the levels move now
            level[0] += gamma * (LOWER_Q - (rows["actual"] < lower).mean())
            level[1] += gamma * ((rows["actual"] > upper).mean() - (1 - UPPER_Q))
        parts.append(today)
    if not parts:
        return pd.DataFrame(columns=[*df.columns.drop(["score", "day", "group"]), "lower", "upper"])
    return pd.concat(parts, ignore_index=True).drop(columns=["score", "day", "group"])


# Coverage per calendar month and how far it is from 80% on average and at worst
def monthly_coverage(df):
    inside = (df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])
    by_month = inside.groupby(df["time_local"].dt.strftime("%Y-%m")).mean()
    gap = (by_month - TARGET).abs()
    return {"month_gap": float(gap.mean()), "worst_month": float(by_month.iloc[gap.argmax()])}


def run_version(version, preds, features, start, end):
    if "gamma" not in version:
        return add_intervals(preds, features, "constant", start, end)
    return adaptive_intervals(preds, features, start, end, version["gamma"], version["mondrian"])


# Every version is measured on exactly the same hours as the current ranges
def compare_versions(preds, features, start, end):
    results = {v["name"]: run_version(v, preds, features, start, end) for v in VERSIONS}
    hours = set(results["constant (current)"]["time_utc"])
    rows = []
    for name, df in results.items():
        df = df[df["time_utc"].isin(hours)]
        rows.append({"version": name, **interval_metrics(df), **monthly_coverage(df)})
    return pd.DataFrame(rows)


# The rule, fixed before looking at the results: the goal of 14D is coverage that stays close to
# 80% month after month, so the version with the smallest average monthly gap wins. The interval
# score is shown next to it, so a version that only wins by being very wide would be visible.
def choose(result):
    return result.sort_values(["month_gap", "interval_score"]).reset_index(drop=True)


# Saves the test-year errors (actual minus forecast) that the live ranges start from
def save_scores():
    test = pd.read_parquet(TEST_PREDICTIONS)
    scores = pd.DataFrame({"time_utc": test["time_utc"], "score": (test["actual"] - test["prediction"]).round(6)})
    scores.to_csv(SCORES, index=False)
    return scores


# The adaptive range for the hours of `forecast` (the next day), the same way as in 14D.1.
# Errors: the saved test-year errors plus every earlier live hour with a known price. The levels are
# replayed from the first live day on every run, so no state has to be stored and a repeated run
# gives the same answer. Returns lower and upper in the order of `forecast` (empty if not possible).
def live_ranges(record, forecast):
    saved = pd.read_csv(SCORES)
    saved["time_utc"] = pd.to_datetime(saved["time_utc"], utc=True)
    day_start = forecast["time_local"].min()
    if record is None:  # the very first run has no live record yet
        record = forecast.iloc[:0].assign(actual=np.nan)
    # Only days before the forecast day: a run after 13:00 already knows the next day's prices
    done = record[record["actual"].notna() & (record["time_local"] < day_start)]
    # A saved error is used as a forecast of 0 with the error as the price, which gives the same score
    preds = pd.concat([
        pd.DataFrame({"time_utc": saved["time_utc"], "prediction": 0.0, "actual": saved["score"]}),
        pd.DataFrame({"time_utc": done["time_utc"], "prediction": done["lstm"], "actual": done["actual"]}),
        pd.DataFrame({"time_utc": forecast["time_utc"], "prediction": forecast["lstm"], "actual": np.nan}),
    ], ignore_index=True).drop_duplicates("time_utc", keep="last")
    features = pd.DataFrame({"time_utc": preds["time_utc"], "time_local": preds["time_utc"].dt.tz_convert(TZ)})
    start = done["time_local"].min().normalize() if len(done) else day_start
    end = forecast["time_local"].max() + pd.Timedelta(hours=1)
    ranges = adaptive_intervals(preds, features, start, end, CHOSEN["gamma"], CHOSEN["mondrian"])
    if ranges.empty:  # fewer than MIN_CAL_DAYS days of errors
        return forecast[["time_utc"]].assign(lower=np.nan, upper=np.nan)
    return forecast[["time_utc"]].merge(ranges[["time_utc", "lower", "upper"]], on="time_utc", how="left")


def main():
    if sys.argv[1:] == ["scores"]:
        scores = save_scores()
        print(f"saved {len(scores)} test-year errors from {scores['time_utc'].min()} to {scores['time_utc'].max()}")
        return
    features = pd.read_parquet(FEATURES)
    valid = pd.read_parquet(VALID_PREDICTIONS)
    result = choose(compare_versions(valid, features, VALID_START, TEST_START))
    result.to_csv(RESULTS / "intervals_adaptive_validation.csv", index=False)
    print("validation year, the same hours for every version:")
    print(result[["version", "hours", "coverage", "month_gap", "worst_month", "width", "interval_score",
                  "coverage_top5"]].round(4).to_string(index=False))
    print(f"chosen: {result['version'].iloc[0]} (smallest average monthly gap from 80%)")


if __name__ == "__main__":
    main()