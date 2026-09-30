# Tests for weather_forecast.py: filling small gaps and swapping in the forecast weather.

import numpy as np
import pandas as pd
from se3_electricity_forecast.weather_forecast import WEATHER_COLS, fill_small_gaps, use_forecast_weather


def test_fill_small_gaps_fills_at_most_3_hours():
    values = [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan, 11.0]
    df = pd.DataFrame({col: values for col in WEATHER_COLS})
    filled, missing = fill_small_gaps(df.copy())
    assert missing == 7 * len(WEATHER_COLS)  # all missing values are counted and reported
    assert filled["temperature_2m"].iloc[1:3].tolist() == [2.0, 3.0]  # a 2-hour gap is filled
    assert filled["temperature_2m"].isna().sum() == 2  # only 3 hours of the 5-hour gap are filled


def test_use_forecast_weather_swaps_only_the_weather(prices, weather):
    features = prices.merge(weather, on="time_utc")
    forecast = weather.copy()
    forecast[WEATHER_COLS] = forecast[WEATHER_COLS] + 100
    swapped = use_forecast_weather(features, forecast)
    assert len(swapped) == len(features)
    assert (swapped[WEATHER_COLS].to_numpy() == forecast[WEATHER_COLS].to_numpy()).all()
    assert (swapped["price_sek_kwh"].to_numpy() == features["price_sek_kwh"].to_numpy()).all()