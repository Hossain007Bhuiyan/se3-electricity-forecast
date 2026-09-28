# Builds the feature table the models learn from and saves it to data/processed/features.parquet.
# Every row is one hour we want to predict. Price features only use data up to the day
# before, because that is all we know in the morning when the forecast is made.


from datetime import timedelta
from pathlib import Path
import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Returns a set of all Swedish holiday dates for the given years.
def swedish_holidays(years):
    # include_sundays=False: the package counts every Sunday as a holiday by default.
    # language="sv": holiday names follow the computer's language otherwise, and the
    # "Midsommardagen" check below would fail on an English system.
    se = holidays.Sweden(years=years, include_sundays=False, language="sv")
    days = set(se.keys())

    # Midsommarafton, Christmas Eve and New Year's Eve are not official holidays,
    # but most workplaces are closed, so electricity demand behaves like a holiday
    for day, name in se.items():
        if name == "Midsommardagen":
            days.add(day - timedelta(days=1))
    for year in years:
        days.add(pd.Timestamp(year, 12, 24).date())
        days.add(pd.Timestamp(year, 12, 31).date())
    return days

# Adds hour, weekday, month, weekend and holiday columns based on Swedish local time.
def add_calendar_features(df):
    local = df["time_local"]
    df["hour"] = local.dt.hour
    df["weekday"] = local.dt.dayofweek
    df["month"] = local.dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    # +2 so next year's holidays are also known when forecasting across New Year
    years = range(local.dt.year.min(), local.dt.year.max() + 2)
    df["is_holiday"] = local.dt.date.isin(swedish_holidays(years)).astype(int)
    return df


# Adds the price from the same hour 1 day, 2 days and 1 week earlier.
# shift(24) moves the column down 24 rows. This equals 24 hours only because
# the data has no missing hours (checked in the notebooks).
def add_lag_features(df):
    price = df["price_sek_kwh"]
    df["price_lag_24h"] = price.shift(24)
    df["price_lag_48h"] = price.shift(48)
    df["price_lag_168h"] = price.shift(168)
    return df

# Adds statistics of the previous day (mean, min, max, std) and the average of the last 7 days.
def add_previous_day_features(df):
    df["date"] = df["time_local"].dt.date
    daily = df.groupby("date")["price_sek_kwh"].agg(["mean", "min", "max", "std"])
    daily["mean_7d"] = daily["mean"].rolling(7).mean()
    # Move everything one day forward, so each day gets the statistics of the day
    # before and never of itself. This line is what prevents leakage here.
    daily = daily.shift(1)
    daily.columns = ["prev_day_mean", "prev_day_min", "prev_day_max", "prev_day_std", "prev_7d_mean"]
    df = df.merge(daily, left_on="date", right_index=True, how="left")
    return df.drop(columns=["date"])


def build_features(prices, weather):
    df = prices.sort_values("time_utc").reset_index(drop=True)
    # The euro price is the same as the target in another currency, so the model must never see it
    df = df.drop(columns=["price_eur_kwh"])
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_previous_day_features(df)
    df = df.merge(weather, on="time_utc", how="left")
    # Drops the first week (no week-old prices yet) and the last hours (no weather yet)
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