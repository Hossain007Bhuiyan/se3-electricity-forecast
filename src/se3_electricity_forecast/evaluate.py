# Shared evaluation tools: the time-based split and the error measures.
# Every model imports these, so all results are calculated on the same hours in the same way.

import numpy as np
import pandas as pd

# Split dates in Swedish local time, so each period starts exactly at midnight.
# Train: before VALID_START. Validation: one year. Test: from TEST_START until TEST_END.
TZ = "Europe/Stockholm"
VALID_START = pd.Timestamp("2024-10-01", tz=TZ)
TEST_START = pd.Timestamp("2025-10-01", tz=TZ)
# Fixed end, so test results stay the same when new data is downloaded later
TEST_END = pd.Timestamp("2026-09-27", tz=TZ)


# Splits the table by date, never randomly, so no model can learn from the future.
def split(df):
    t = df["time_local"]
    train = df[t < VALID_START]
    valid = df[(t >= VALID_START) & (t < TEST_START)]
    test = df[(t >= TEST_START) & (t < TEST_END)]
    return train, valid, test


# Returns the month start dates between start and end, e.g. 1 Oct 2024, 1 Nov 2024, ...
# Used by the monthly walk-forward training. If the period does not end on a month start
# (like the test period), the end date is added as the last boundary.
def month_bounds(start, end):
    bounds = list(pd.date_range(start, end, freq="MS"))
    if bounds[-1] < end:
        bounds.append(end)
    return bounds


# Mean absolute error: the average size of the error in SEK/kWh.
# np.asarray compares values by position instead of by pandas row labels.
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# Root mean squared error: like MAE, but big errors (price spikes) count more.
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))