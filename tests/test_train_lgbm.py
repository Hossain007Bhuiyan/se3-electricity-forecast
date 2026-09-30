# Tests for train_lgbm.py (LightGBM). PyTorch is never imported in this folder, because LightGBM
# and PyTorch must not run in the same Python process on macOS (see final_test.py).

import sys
import pandas as pd
from se3_electricity_forecast import train_lgbm
from se3_electricity_forecast.features import build_features

TZ = "Europe/Stockholm"


def test_pytorch_is_not_loaded():
    # No test in this folder may load PyTorch, since LightGBM runs here (the OpenMP rule from Step 6C)
    assert "torch" not in sys.modules


def test_diff_target_turns_back_into_the_price(prices, weather):
    df = build_features(prices, weather)
    target = train_lgbm.make_target(df, "diff")
    restored = train_lgbm.to_price(target.to_numpy(), df, "diff")
    assert abs(restored - df["price_sek_kwh"].to_numpy()).max() < 1e-12


def test_walk_forward_predicts_every_hour_once(prices, weather):
    df = build_features(prices, weather)
    # Walk-forward periods always start on the first day of a month, like VALID_START and TEST_START
    start, end = pd.Timestamp("2025-11-01", tz=TZ), pd.Timestamp("2025-11-05", tz=TZ)
    config = next(c for c in train_lgbm.CONFIGS if c["name"] == "lgbm_price_l1")
    preds = train_lgbm.walk_forward(df, df, config, start, end)
    expected = df[(df["time_local"] >= start) & (df["time_local"] < end)]
    assert len(preds) == len(expected)
    assert preds["time_utc"].is_unique
    assert set(preds["time_utc"]) == set(expected["time_utc"])
    assert (preds["actual"].to_numpy() == expected["price_sek_kwh"].to_numpy()).all()


def test_mlflow_params_describe_the_version():
    config = next(c for c in train_lgbm.CONFIGS if c["name"] == "lgbm_price_l1")
    params = train_lgbm.mlflow_params(config, pd.Timestamp("2024-10-01", tz=TZ), pd.Timestamp("2025-10-01", tz=TZ))
    assert params["objective"] == "l1" and params["target"] == "price"
    assert params["period_start"] == "2024-10-01" and params["period_end"] == "2025-10-01"