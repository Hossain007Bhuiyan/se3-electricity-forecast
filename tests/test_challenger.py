# Tests for challenger.py. Small made-up tables, so they run anywhere in a few seconds.
# PyTorch is never imported here, because LightGBM runs in this folder (see test_train_lgbm.py).

import sys
import numpy as np
import pandas as pd
from se3_electricity_forecast import challenger
from se3_electricity_forecast.features import FEATURE_COLS, build_features
from se3_electricity_forecast.market_features import MARKET_COLS

TZ = "Europe/Stockholm"


def test_pytorch_is_not_loaded():
    assert "torch" not in sys.modules


def test_only_the_challenger_gets_the_market_inputs():
    configs = {c["name"]: c for c in challenger.CONFIGS}
    assert challenger.columns(configs["lgbm_price_l1"]) == FEATURE_COLS
    assert challenger.columns(configs["challenger_l1"]) == FEATURE_COLS + MARKET_COLS


def test_walk_forward_predicts_every_hour_once(prices, weather):
    df = build_features(prices, weather)
    rng = np.random.default_rng(1)
    for col in MARKET_COLS:
        df[col] = rng.random(len(df))
    start, end = pd.Timestamp("2025-11-01", tz=TZ), pd.Timestamp("2025-11-05", tz=TZ)
    config = next(c for c in challenger.CONFIGS if c["name"] == "challenger_l1")
    preds = challenger.walk_forward(df, df, config, start, end)
    expected = df[(df["time_local"] >= start) & (df["time_local"] < end)]
    assert len(preds) == len(expected) and preds["time_utc"].is_unique
    assert (preds["actual"].to_numpy() == expected["price_sek_kwh"].to_numpy()).all()


def test_average_matches_hours_and_summary_compares_with_the_baseline():
    times = pd.date_range("2025-10-01", periods=3, freq="h", tz="UTC")
    a = pd.DataFrame({"time_utc": times, "actual": [1.0, 2.0, 3.0], "prediction": [1.0, 2.0, 5.0]})
    b = pd.DataFrame({"time_utc": times[::-1], "actual": [3.0, 2.0, 1.0], "prediction": [3.0, 4.0, 3.0]})
    both = challenger.average(a, b)
    assert both["prediction"].tolist() == [2.0, 3.0, 4.0]  # matched by hour, not by row order
    table = challenger.summary({"a": a, "both": both}, naive_mae=2.0).set_index("model")
    assert np.isclose(table.loc["a", "mae"], 2 / 3) and np.isclose(table.loc["a", "rel_mae"], 1 / 3)
    assert np.isclose(table.loc["both", "mae"], 1.0)


def test_mlflow_params_describe_the_challenger():
    config = next(c for c in challenger.CONFIGS if c["name"] == "challenger_l2")
    params = challenger.mlflow_params(config)
    assert params["objective"] == "l2" and params["features"].split(",") == FEATURE_COLS + MARKET_COLS
    assert params["period_start"] == "2024-10-01" and params["period_end"] == "2025-10-01"