from datetime import timedelta
from pathlib import Path

import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def swedish_holidays(years):
    se = holidays.Sweden(years=years, include_sundays=False, language="sv")
    days = set(se.keys())
    for day, name in se.items():
        if name == "Midsommardagen":
            days.add(day - timedelta(days=1))
    for year in years:
        days.add(pd.Timestamp(year, 12, 24).date())
        days.add(pd.Timestamp(year, 12, 31).date())
    return days


def add_calendar_features(df):
    local = df["time_local"]
    df["hour"] = local.dt.hour
    df["weekday"] = local.dt.dayofweek
    df["month"] = local.dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    years = range(local.dt.year.min(), local.dt.year.max() + 2)
    df["is_holiday"] = local.dt.date.isin(swedish_holidays(years)).astype(int)
    return df


def add_lag_features(df):
    price = df["price_sek_kwh"]
    df["price_lag_24h"] = price.shift(24)
    df["price_lag_48h"] = price.shift(48)
    df["price_lag_168h"] = price.shift(168)
    return df


def add_previous_day_features(df):
    df["date"] = df["time_local"].dt.date
    daily = df.groupby("date")["price_sek_kwh"].agg(["mean", "min", "max", "std"])
    daily["mean_7d"] = daily["mean"].rolling(7).mean()
    daily = daily.shift(1)
    daily.columns = ["prev_day_mean", "prev_day_min", "prev_day_max", "prev_day_std", "prev_7d_mean"]
    df = df.merge(daily, left_on="date", right_index=True, how="left")
    return df.drop(columns=["date"])


def build_features(prices, weather):
    df = prices.sort_values("time_utc").reset_index(drop=True)
    df = df.drop(columns=["price_eur_kwh"])
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_previous_day_features(df)
    df = df.merge(weather, on="time_utc", how="left")
    return df.dropna().reset_index(drop=True)


def main():
    prices = pd.read_parquet(RAW / "prices_se3_hourly.parquet")
    weather = pd.read_parquet(RAW / "weather_stockholm_hourly.parquet")

    features = build_features(prices, weather)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED / "features.parquet", index=False)

    print("rows:", len(features))
    print("from", features["time_local"].min(), "to", features["time_local"].max())
    print("columns:", features.columns.tolist())


if __name__ == "__main__":
    main()