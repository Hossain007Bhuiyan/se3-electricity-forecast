# Tests for baselines.py: the four rule-based forecasts.

import pandas as pd
from se3_electricity_forecast.baselines import baseline_predictions


def test_weekly_naive_uses_last_week_on_monday_saturday_and_sunday():
    df = pd.DataFrame({
        "weekday": list(range(7)),  # 0 = Monday ... 6 = Sunday
        "price_lag_24h": [1.0] * 7,
        "price_lag_168h": [2.0] * 7,
        "prev_day_mean": [3.0] * 7,
    })
    preds = baseline_predictions(df)
    assert list(preds["weekly_naive"]) == [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    assert list(preds["same_hour_yesterday"]) == [1.0] * 7
    assert list(preds["same_hour_last_week"]) == [2.0] * 7
    assert list(preds["yesterday_mean"]) == [3.0] * 7