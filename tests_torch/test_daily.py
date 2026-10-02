# Tests for daily.py, the daily live forecast. No internet is used: the downloads are replaced
# by synthetic data, and all files are written to a temporary folder.

from datetime import date

import numpy as np
import pandas as pd
import pytest

from se3_electricity_forecast import daily, train_lstm
from se3_electricity_forecast.features import FEATURE_COLS, build_features

TZ = "Europe/Stockholm"
WEATHER_COLS = ["temperature_2m", "wind_speed_10m", "precipitation", "cloud_cover", "shortwave_radiation"]


def make_prices(start, end):
    time_utc = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ), freq="h",
                             inclusive="left").tz_convert("UTC")
    price = 1 + np.sin(np.arange(len(time_utc)) / 5) + np.arange(len(time_utc)) * 0.0001
    return pd.DataFrame({"time_utc": time_utc, "price_sek_kwh": price, "price_eur_kwh": price / 11,
                         "time_local": time_utc.tz_convert(TZ)})


def make_weather(time_utc):
    rng = np.random.default_rng(0)
    weather = pd.DataFrame({"time_utc": time_utc})
    for col in WEATHER_COLS:
        weather[col] = rng.random(len(time_utc))
    return weather


# Points every file daily.py writes to a temporary folder
@pytest.fixture
def folders(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "DATA", tmp_path / "data")
    monkeypatch.setattr(daily, "LIVE", tmp_path / "live")
    monkeypatch.setattr(daily, "PRICES", tmp_path / "data" / "prices.parquet")
    monkeypatch.setattr(daily, "MODEL", tmp_path / "live" / "model" / "lstm.pt")
    monkeypatch.setattr(daily, "RECORD", tmp_path / "live" / "forecasts.csv")
    monkeypatch.setattr(daily, "SUMMARY", tmp_path / "live" / "summary.json")
    return tmp_path


def test_day_features_match_the_training_features():
    # The live features of a day must be exactly the features the model was trained with,
    # even though the live version does not know that day's prices
    prices = make_prices("2025-10-01", "2025-11-05")
    weather = make_weather(prices["time_utc"])
    live = daily.day_features(prices, weather, date(2025, 10, 20))
    training = build_features(prices, weather)
    training = training[training["time_local"].dt.date == date(2025, 10, 20)].reset_index(drop=True)
    pd.testing.assert_frame_equal(live[FEATURE_COLS], training[FEATURE_COLS])
    assert live["price_sek_kwh"].isna().all()  # the day's own prices are never used


def test_day_features_cover_the_25_hour_day():
    prices = make_prices("2025-10-01", "2025-11-05")
    rows = daily.day_features(prices, make_weather(prices["time_utc"]), date(2025, 10, 26))
    assert len(rows) == 25  # 26 Oct 2025: the clock went back one hour


def test_day_features_need_the_complete_day_before():
    prices = make_prices("2025-10-01", "2025-11-05")
    incomplete = prices[prices["time_local"] < pd.Timestamp("2025-10-19 23:00", tz=TZ)]
    with pytest.raises(ValueError):
        daily.day_features(incomplete, make_weather(prices["time_utc"]), date(2025, 10, 20))


def test_model_is_saved_and_loaded_exactly(folders, monkeypatch):
    monkeypatch.setattr(train_lstm, "MAX_EPOCHS", 1)
    prices = make_prices("2025-10-01", "2025-11-05")
    features = build_features(prices, make_weather(prices["time_utc"]))
    df = train_lstm.add_sequence_end(train_lstm.add_cyclical(features), prices)
    price_values = prices["price_sek_kwh"].to_numpy()
    model, stats, epochs = train_lstm.fit(df.iloc[:400], df.iloc[400:500], price_values)
    daily.save_model(model, stats, "2025-11", epochs)

    loaded, loaded_stats, month = daily.load_model()
    rows = df.iloc[500:548]
    assert month == "2025-11"
    assert np.array_equal(train_lstm.predict_prices(model, stats, rows, price_values),
                          train_lstm.predict_prices(loaded, loaded_stats, rows, price_values))


def test_record_keeps_the_first_forecast_and_fills_real_prices():
    prices = make_prices("2025-10-01", "2025-10-05")
    hours = prices.iloc[24:48]
    first = pd.DataFrame({"time_utc": hours["time_utc"], "time_local": hours["time_local"],
                          "issued_at_utc": pd.Timestamp("2025-10-01 05:00", tz="UTC"),
                          "issued_before_noon": True, "model_month": "2025-10",
                          "lstm": hours["price_sek_kwh"] + 0.1, "weekly_naive": hours["price_sek_kwh"] + 0.2})
    second = first.assign(lstm=first["lstm"] + 5)  # a later run for the same hours

    record = daily.update_record(None, first, prices)
    record = daily.update_record(record, second, prices)
    assert len(record) == 24                                    # no duplicate hours
    assert np.allclose(record["lstm"], hours["price_sek_kwh"] + 0.1)  # the first forecast counts
    assert np.allclose(record["actual"], hours["price_sek_kwh"])

    summary = daily.summarize(record)
    assert summary["hours"] == 24 and summary["days"] == 1
    assert np.isclose(summary["lstm_mae"], 0.1) and np.isclose(summary["rel_mae"], 0.5)


def test_full_daily_run_without_internet(folders, monkeypatch):
    monkeypatch.setattr(train_lstm, "MAX_EPOCHS", 1)
    prices = make_prices("2025-01-01", "2025-10-21")  # known up to the end of 20 Oct
    weather = make_weather(make_prices("2025-01-01", "2025-10-24")["time_utc"])
    monkeypatch.setattr(daily, "update_prices", lambda day: prices)
    monkeypatch.setattr(daily.data_download, "download_weather", lambda start, end: weather)
    monkeypatch.setattr(daily, "download_weather_forecast", lambda: weather)

    daily.main(now=pd.Timestamp("2025-10-20 05:30", tz="UTC"))  # 07:30 Swedish time on 20 Oct
    record = daily.read_record()
    assert len(record) == 24
    assert (record["time_local"].dt.date == date(2025, 10, 21)).all()
    assert record["issued_before_noon"].all() and record["actual"].isna().all()
    assert daily.load_model()[2] == "2025-10"

    # Running again for the same day must not add or change anything
    daily.main(now=pd.Timestamp("2025-10-20 06:30", tz="UTC"))
    assert daily.read_record()["lstm"].equals(record["lstm"])