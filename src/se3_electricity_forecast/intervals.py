# Prediction intervals for the LSTM forecasts. Instead of one number per hour, every forecast gets
# an 80% range: the real price should fall inside it in about 8 of 10 hours.
#
# Method: split conformal prediction with a scale. The LSTM's own errors on earlier months (from
# forecasts it made without knowing those prices) are divided by a scale that is known in the
# morning, for example how much the prices moved yesterday. The 10% and 90% quantiles of these
# scaled errors give the range:
#     lower = forecast + scale * q10        upper = forecast + scale * q90
# After calm days the range is narrow and after volatile days it is wide. It can also be uneven,
# because prices more often jump far up than far down. No new model is trained.
#
# Only errors from before the month in question are used, so a range never uses information from
# the future. The scale is chosen on the validation year; the test year is used once at the end.
#     uv run python -m se3_electricity_forecast.intervals         compare the scales on the validation year
#     uv run python -m se3_electricity_forecast.intervals test    test year with the chosen scale
# Only pandas and NumPy are used here, no model library.

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from se3_electricity_forecast.evaluate import TEST_END, TEST_START, VALID_START, month_bounds

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "features.parquet"
VALID_PREDICTIONS = ROOT / "data" / "processed" / "lstm_valid_predictions.parquet"
TEST_PREDICTIONS = ROOT / "data" / "processed" / "test_predictions_lstm.parquet"
RESULTS = ROOT / "results"
CALIBRATION = RESULTS / "interval_calibration.json"

LOWER_Q, UPPER_Q = 0.1, 0.9   # an 80% range
CAL_DAYS = 365                # at most the errors of the last year are used
MIN_CAL_DAYS = 28             # a month is skipped if fewer earlier days with errors exist
SCALE_FLOOR = 0.01            # the smallest scale in SEK/kWh, so a very calm day never gives a zero range

# The scales that are compared. All of them are known on the morning before the forecast day.
#   constant:      the same range for every hour, the usual split conformal prediction
#   prev_day_std:  wider after a day with large price swings
#   prev_day_mean: wider after an expensive day (errors in SEK/kWh grow with the price level)
SCALES = ["constant", "prev_day_std", "prev_day_mean"]


def scale_values(df, scale):
    if scale == "constant":
        return np.ones(len(df))
    return np.maximum(np.abs(df[scale].to_numpy(float)), SCALE_FLOOR)


# Adds the 80% range to the forecasts of every month from start to end. `preds` holds forecasts
# made without knowing the prices (validation and test predictions), including earlier months,
# whose errors are used for calibration. Months without enough earlier errors are left out.
def add_intervals(preds, features, scale, start, end):
    df = preds.merge(features[["time_utc", "time_local", "prev_day_std", "prev_day_mean"]], on="time_utc")
    df["scale"] = scale_values(df, scale)
    df["score"] = (df["actual"] - df["prediction"]) / df["scale"]
    parts = []
    for month_start, month_end in zip(month_bounds(start, end)[:-1], month_bounds(start, end)[1:]):
        earlier = df[(df["time_local"] < month_start) & (df["time_local"] >= month_start - pd.Timedelta(days=CAL_DAYS))]
        if earlier["time_local"].dt.date.nunique() < MIN_CAL_DAYS:
            continue
        q_low, q_high = earlier["score"].quantile([LOWER_Q, UPPER_Q])
        month = df[(df["time_local"] >= month_start) & (df["time_local"] < month_end)].copy()
        month["lower"] = month["prediction"] + month["scale"] * q_low
        month["upper"] = month["prediction"] + month["scale"] * q_high
        parts.append(month)
    return pd.concat(parts, ignore_index=True)


def pinball(q, actual, forecast):
    diff = actual - forecast
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# Coverage: share of hours inside the range (should be close to 0.80). Width: average size of the
# range. Pinball loss and interval score reward ranges that are narrow and still contain the price,
# and punish misses by how far they are off. Lower is better for both.
def interval_metrics(df):
    actual, lower, upper = df["actual"].to_numpy(), df["lower"].to_numpy(), df["upper"].to_numpy()
    alpha = 1 - (UPPER_Q - LOWER_Q)
    score = (upper - lower) + (2 / alpha) * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))
    spikes = actual >= np.quantile(actual, 0.95)
    inside = (actual >= lower) & (actual <= upper)
    return {
        "hours": int(len(df)),
        "coverage": float(inside.mean()),
        "coverage_top5": float(inside[spikes].mean()),
        "width": float(np.mean(upper - lower)),
        "pinball": (pinball(LOWER_Q, actual, lower) + pinball(UPPER_Q, actual, upper)) / 2,
        "interval_score": float(np.mean(score)),
    }


def compare_scales(preds, features, start, end):
    rows = []
    for scale in SCALES:
        rows.append({"scale": scale, **interval_metrics(add_intervals(preds, features, scale, start, end))})
    return pd.DataFrame(rows).sort_values("interval_score").reset_index(drop=True)


# The quantiles for live use: from the scaled errors of the last CAL_DAYS days before `end`
def calibration(preds, features, scale, end):
    df = preds.merge(features[["time_utc", "time_local", "prev_day_std", "prev_day_mean"]], on="time_utc")
    df = df[(df["time_local"] < end) & (df["time_local"] >= end - pd.Timedelta(days=CAL_DAYS))]
    score = (df["actual"] - df["prediction"]) / scale_values(df, scale)
    q_low, q_high = score.quantile([LOWER_Q, UPPER_Q])
    return {"scale": scale, "q_low": float(q_low), "q_high": float(q_high), "lower_q": LOWER_Q, "upper_q": UPPER_Q,
            "calibrated_on": [str(df["time_local"].min().date()), str(df["time_local"].max().date())]}


def main():
    features = pd.read_parquet(FEATURES)
    valid = pd.read_parquet(VALID_PREDICTIONS)
    if sys.argv[1:] == ["test"]:
        # The scale was chosen on the validation year; the first row of that file is the best one
        scale = pd.read_csv(RESULTS / "intervals_validation.csv")["scale"].iloc[0]
        preds = pd.concat([valid, pd.read_parquet(TEST_PREDICTIONS)], ignore_index=True)
        result = pd.DataFrame([{"scale": scale, **interval_metrics(add_intervals(preds, features, scale, TEST_START, TEST_END))}])
        result.to_csv(RESULTS / "intervals_test.csv", index=False)
        CALIBRATION.write_text(json.dumps(calibration(preds, features, scale, TEST_END), indent=2) + "\n")
        print(f"test year, scale {scale}:")
        print(result.round(4).to_string(index=False))
        print(f"saved {CALIBRATION.relative_to(ROOT)} for the live forecasts")
        return
    result = compare_scales(valid, features, VALID_START, TEST_START)
    result.to_csv(RESULTS / "intervals_validation.csv", index=False)
    print("validation year (the first month has no earlier errors, so it is left out):")
    print(result.round(4).to_string(index=False))
    print(f"chosen scale: {result['scale'].iloc[0]} (lowest interval score)")


if __name__ == "__main__":
    main()