# Tests for the helper functions in explain.py.

import numpy as np
import pandas as pd
from se3_electricity_forecast.explain import colour_range, permutation_importance


def test_colour_range_follows_the_shap_rule():
    continuous = np.arange(101, dtype=float)
    assert colour_range(continuous) == (5.0, 95.0)  # 5th to 95th percentile
    mostly_zero = np.array([0.0] * 97 + [1.0] * 3)
    assert colour_range(mostly_zero) == (0.0, 1.0)  # falls back so 0 and 1 still get two colours


def test_permutation_importance_finds_the_input_the_model_uses():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"a": rng.random(500), "b": rng.random(500)})
    data["price_sek_kwh"] = 2 * data["a"]

    def predict_fn(df):
        return 2 * df["a"].to_numpy()  # a perfect model that only uses "a"

    base_mae, importance = permutation_importance(predict_fn, data, {"a": ["a"], "b": ["b"]})
    scores = importance.set_index("feature")["mae_increase"]
    assert base_mae == 0.0
    assert scores["a"] > 0.1   # shuffling "a" makes the model much worse
    assert scores["b"] == 0.0  # the model never uses "b"