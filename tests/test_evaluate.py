# Tests for evaluate.py: the time-based split, the error measures and the monthly boundaries.

import numpy as np
import pandas as pd
from se3_electricity_forecast.evaluate import (
    TEST_END, TEST_START, TZ, VALID_START, mae, month_bounds, rmse, split,
)


def test_split_is_by_time_and_has_the_expected_sizes():
    time_local = pd.date_range(pd.Timestamp("2024-09-01", tz=TZ), pd.Timestamp("2026-09-28", tz=TZ),
                               freq="h", inclusive="left")
    train, valid, test = split(pd.DataFrame({"time_local": time_local}))
    # No overlap and no gaps between the parts
    assert train["time_local"].max() < VALID_START <= valid["time_local"].min()
    assert valid["time_local"].max() < TEST_START <= test["time_local"].min()
    assert test["time_local"].max() < TEST_END
    # One full year for validation, and the test year up to 26 Sep 2026 (the same as in Step 5)
    assert len(valid) == 8760
    assert len(test) == 8664


def test_mae_and_rmse():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.0, 3.0, 1.0, 4.0])
    assert mae(actual, predicted) == 0.75
    assert rmse(actual, predicted) == np.sqrt(5 / 4)


def test_mae_compares_by_position_not_by_label():
    # Two pandas Series with different row labels must still be compared position by position
    actual = pd.Series([1.0, 2.0], index=[10, 11])
    predicted = pd.Series([1.0, 2.0], index=[0, 1])
    assert mae(actual, predicted) == 0.0


def test_month_bounds():
    valid = month_bounds(VALID_START, TEST_START)
    assert len(valid) == 13  # 12 months
    assert valid[0] == VALID_START and valid[-1] == TEST_START
    test = month_bounds(TEST_START, TEST_END)
    assert len(test) == 13  # 12 months, the last one ending on TEST_END
    assert test[-1] == TEST_END