# Tests for challenger_live.py. ENTSO-E, Open-Meteo and the model are faked, so no internet,
# no token and no trained model are needed.

import sys
import numpy as np
import pandas as pd
import pytest
from se3_electricity_forecast import challenger_live
from se3_electricity_forecast.features import FEATURE_COLS
from se3_electricity_forecast.market_features import WIND_COLS
from test_market_features import make_entsoe

TZ = "Europe/Stockholm"
DAY = pd.Timestamp("2026-10-08").date()  # the forecast day; the run day is 7 Oct
NOW = pd.Timestamp("2026-10-07 09:30", tz="UTC")  # 11:30 Swedish time


def hours_of(day):
    times = pd.date_range(pd.Timestamp(day, tz=TZ), periods=24, freq="h")
    return pd.DataFrame({"time_utc": times.tz_convert("UTC"), "time_local": times})


# The LSTM's record and the saved inputs for the forecast day, as the daily run writes them
@pytest.fixture
def live():
    record = hours_of(DAY).assign(lstm=1.0)
    inputs = hours_of(DAY).assign(**{col: 0.5 for col in FEATURE_COLS})
    return record, inputs


def fake_fetch(day):
    reservoir = pd.DataFrame({"week_start_utc": pd.to_datetime(["2026-09-20 22:00"], utc=True), "reservoir_SE": [25e6]})
    wind = hours_of(day)[["time_utc"]].assign(**{col: 10.0 for col in WIND_COLS})
    return make_entsoe(), reservoir, wind


class FakeModel:
    def predict(self, rows):
        assert list(rows.columns) == challenger_live.COLS  # the inputs in the order the model was trained with
        return np.full(len(rows), 2.0)


def test_pytorch_is_not_loaded():
    assert "torch" not in sys.modules


def test_forecast_is_the_average_of_the_lstm_and_the_challenger(live):
    record, inputs = live
    forecast = challenger_live.make_forecast(record, inputs, DAY, NOW, FakeModel(), fetch=fake_fetch)
    assert len(forecast) == 24 and not forecast["fallback"].any()
    assert (forecast["challenger"] == 2.0).all() and (forecast["combined"] == 1.5).all()
    assert forecast["issued_before_noon"].all()


def test_a_missing_market_input_falls_back_to_the_lstm(live):
    record, inputs = live

    def outage(day):
        raise RuntimeError("ENTSO-E answered with HTTP 599")

    forecast = challenger_live.make_forecast(record, inputs, DAY, NOW, FakeModel(), fetch=outage)
    assert forecast["fallback"].all() and (forecast["combined"] == forecast["lstm"]).all()
    assert "HTTP 599" in forecast["note"].iloc[0]

    def no_load_forecast(day):
        entsoe, reservoir, wind = fake_fetch(day)
        entsoe["load_forecast"] = np.nan  # published late, like on 25 October 2026
        return entsoe, reservoir, wind

    late = challenger_live.make_forecast(record, inputs, DAY, NOW, FakeModel(), fetch=no_load_forecast)
    assert late["fallback"].all() and "load_forecast" in late["note"].iloc[0]


def test_a_forecast_after_noon_does_not_count(live):
    record, inputs = live
    late = pd.Timestamp("2026-10-07 10:30", tz="UTC")  # 12:30 Swedish time
    assert not challenger_live.make_forecast(record, inputs, DAY, late, FakeModel(), fetch=fake_fetch)["issued_before_noon"].any()


def test_there_must_be_an_lstm_forecast_first(live):
    record, inputs = live
    with pytest.raises(RuntimeError, match="no forecast"):
        challenger_live.make_forecast(record.iloc[:0], inputs, DAY, NOW, FakeModel(), fetch=fake_fetch)


def test_the_first_forecast_for_each_hour_counts(live):
    record, inputs = live
    first = challenger_live.make_forecast(record, inputs, DAY, NOW, FakeModel(), fetch=fake_fetch)
    later = first.assign(combined=99.0, issued_at_utc=NOW + pd.Timedelta(hours=3))
    saved = challenger_live.update_saved(challenger_live.update_saved(None, first), later)
    assert len(saved) == 24 and (saved["combined"] == 1.5).all()