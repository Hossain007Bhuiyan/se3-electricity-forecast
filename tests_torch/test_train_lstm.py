# Tests for train_lstm.py (PyTorch). Run separately from the other tests:
#     uv run pytest tests_torch

import sys
from datetime import timedelta
import numpy as np
import pandas as pd
from se3_electricity_forecast import train_lstm

TZ = "Europe/Stockholm"


def test_lightgbm_is_not_loaded():
    # The LSTM code must never load LightGBM (the OpenMP rule from Step 6C)
    assert "lightgbm" not in sys.modules


def test_cyclical_columns_lie_on_a_circle(features):
    df = train_lstm.add_cyclical(features)
    for col in train_lstm.CYCLES:
        assert np.allclose(df[f"{col}_sin"] ** 2 + df[f"{col}_cos"] ** 2, 1)


def test_sequence_ends_at_23_on_the_previous_day(prices, features):
    df = train_lstm.add_sequence_end(features, prices)
    end_times = prices["time_local"].to_numpy()[df["seq_end"].to_numpy()]
    for row_time, end_time in zip(df["time_local"], pd.DatetimeIndex(end_times).tz_convert(TZ)):
        # The last known price is 23:00 on the day before the forecast day. This must also hold
        # around 26 Oct 2025 (a 25-hour day) and on the day after it.
        assert end_time.date() == row_time.date() - timedelta(days=1)
        assert end_time.hour == 23


def test_sequences_contain_only_earlier_prices(prices, features):
    df = train_lstm.add_sequence_end(features, prices)
    price_values = prices["price_sek_kwh"].to_numpy()
    seq = train_lstm.make_sequences(price_values, df["seq_end"].to_numpy())
    assert seq.shape == (len(df), train_lstm.SEQ_LEN)
    # The sequence ends with the price at seq_end, and every price in it is from before the row's day
    assert (seq[:, -1] == price_values[df["seq_end"].to_numpy()]).all()
    first_hour_of_day = prices.groupby(prices["time_local"].dt.date).apply(lambda d: d.index.min())
    for seq_end, row_time in zip(df["seq_end"], df["time_local"]):
        assert seq_end < first_hour_of_day[row_time.date()]


def test_training_is_repeatable_and_fit_predict_matches(prices, features, monkeypatch):
    monkeypatch.setattr(train_lstm, "MAX_EPOCHS", 2)  # keep the test fast
    df = train_lstm.add_sequence_end(train_lstm.add_cyclical(features), prices)
    price_values = prices["price_sek_kwh"].to_numpy()
    train, holdout, predict = df.iloc[:400], df.iloc[400:500], df.iloc[500:560]

    first, epochs = train_lstm.fit_predict(train, holdout, predict, price_values)
    second, _ = train_lstm.fit_predict(train, holdout, predict, price_values)
    model, stats, _ = train_lstm.fit(train, holdout, price_values)
    separate = train_lstm.predict_prices(model, stats, predict, price_values)

    assert epochs <= 2
    assert np.array_equal(first, second)    # the same seed gives exactly the same predictions
    assert np.array_equal(first, separate)  # fit_predict = fit + predict_prices
    assert np.isfinite(first).all()


def test_mlflow_params_describe_the_lstm():
    params = train_lstm.mlflow_params(pd.Timestamp("2024-10-01", tz=TZ), pd.Timestamp("2025-10-01", tz=TZ))
    assert params["seq_len"] == 168 and params["seed"] == 42 and params["loss"] == "L1"
    assert params["period_start"] == "2024-10-01"