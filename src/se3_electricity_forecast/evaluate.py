import numpy as np
import pandas as pd

TZ = "Europe/Stockholm"
VALID_START = pd.Timestamp("2024-10-01", tz=TZ)
TEST_START = pd.Timestamp("2025-10-01", tz=TZ)
TEST_END = pd.Timestamp("2026-09-27", tz=TZ)


def split(df):
    t = df["time_local"]
    train = df[t < VALID_START]
    valid = df[(t >= VALID_START) & (t < TEST_START)]
    test = df[(t >= TEST_START) & (t < TEST_END)]
    return train, valid, test


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))