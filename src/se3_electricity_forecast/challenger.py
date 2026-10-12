# Step 15C.2: a first version of the challenger. LightGBM with the 18 inputs of the current
# models plus the 11 market inputs, trained exactly like the LightGBM in train_lgbm.py: the same fixed
# settings, monthly walk-forward on the validation year, measured weather for training and
# 2-day-old forecasts for predicting. Nothing here uses the test year.
# It is compared with the same LightGBM without market inputs, the LSTM and the average of the
# LSTM and the challenger. This is a first result: the final inputs are decided after the
# availability check (Step 15C.3), so nothing is logged to MLflow yet.
# Run with: uv run python -m se3_electricity_forecast.challenger

from pathlib import Path
import lightgbm as lgb
import pandas as pd
from se3_electricity_forecast.evaluate import TEST_START, TZ, VALID_START, mae, month_bounds, rmse
from se3_electricity_forecast.features import FEATURE_COLS
from se3_electricity_forecast.market_features import MARKET_COLS
from se3_electricity_forecast.train_lgbm import PARAMS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FEATURES = PROCESSED / "features_market.parquet"
FORECAST_FEATURES = PROCESSED / "features_market_forecast_weather.parquet"
LSTM_PREDICTIONS = PROCESSED / "lstm_valid_predictions.parquet"
PREDICTIONS = PROCESSED / "challenger_valid_predictions.parquet"

# lgbm_price_l1 is the best LightGBM from train_lgbm.py, run again here as the fair comparison:
# the same model and data, only without the market inputs
CONFIGS = [
    {"name": "lgbm_price_l1", "objective": "l1", "market": False},
    {"name": "challenger_l1", "objective": "l1", "market": True},
    {"name": "challenger_l2", "objective": "l2", "market": True},
]


def columns(config):
    return FEATURE_COLS + MARKET_COLS if config["market"] else FEATURE_COLS


# For each month: train a new model on everything before that month, then predict the month.
# train_df has measured weather, predict_df has the weather forecasts.
def walk_forward(train_df, predict_df, config, start, end):
    cols = columns(config)
    bounds = month_bounds(start, end)
    parts = []
    for month_start, month_end in zip(bounds[:-1], bounds[1:]):
        train = train_df[train_df["time_local"] < month_start]
        test = predict_df[(predict_df["time_local"] >= month_start) & (predict_df["time_local"] < month_end)]
        model = lgb.LGBMRegressor(objective=config["objective"], **PARAMS)
        model.fit(train[cols], train["price_sek_kwh"])
        parts.append(pd.DataFrame({
            "time_utc": test["time_utc"].to_numpy(),
            "actual": test["price_sek_kwh"].to_numpy(),
            "prediction": model.predict(test[cols]),
        }))
    return pd.concat(parts, ignore_index=True)


# The average of two models' forecasts, matched hour by hour
def average(a, b):
    both = a.merge(b[["time_utc", "prediction"]], on="time_utc", suffixes=("", "_b"))
    both["prediction"] = (both["prediction"] + both.pop("prediction_b")) / 2
    return both


# One row per model: error in SEK/kWh and compared with the weekly_naive baseline (rel_mae)
def summary(preds, naive_mae):
    rows = [{"model": name, "hours": len(p), "mae": mae(p["actual"], p["prediction"]),
             "rmse": rmse(p["actual"], p["prediction"])} for name, p in preds.items()]
    table = pd.DataFrame(rows)
    table["rel_mae"] = table["mae"] / naive_mae
    return table.sort_values("mae").reset_index(drop=True)


# MAE per month, to see if a model is better every month or only in a few
def monthly(preds):
    table = {}
    for name, p in preds.items():
        month = p["time_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m").rename("month")
        table[name] = (p["actual"] - p["prediction"]).abs().groupby(month).mean()
    return pd.DataFrame(table)


def main():
    df = pd.read_parquet(FEATURES)
    df_forecast = pd.read_parquet(FORECAST_FEATURES)
    # The same benchmark as in every earlier step, so rel_mae is directly comparable
    baselines = pd.read_csv(RESULTS / "baselines.csv")
    naive_mae = baselines.query("split == 'valid' and model == 'weekly_naive'")["mae"].iloc[0]

    preds = {}
    for config in CONFIGS:
        print("training", config["name"], "...")
        preds[config["name"]] = walk_forward(df, df_forecast, config, VALID_START, TEST_START)
    # The LSTM's validation forecasts from train_lstm.py and their average with the better challenger
    preds["lstm"] = pd.read_parquet(LSTM_PREDICTIONS)
    best = min(["challenger_l1", "challenger_l2"], key=lambda name: mae(preds[name]["actual"], preds[name]["prediction"]))
    preds[f"lstm + {best}"] = average(preds["lstm"], preds[best])

    pd.concat([p.assign(model=name) for name, p in preds.items()], ignore_index=True).to_parquet(PREDICTIONS, index=False)
    print()
    print(summary(preds, naive_mae).round(4).to_string(index=False))
    print()
    print("MAE per month:")
    print(monthly({name: preds[name] for name in ["lgbm_price_l1", best, "lstm", f"lstm + {best}"]}).round(3).to_string())


if __name__ == "__main__":
    main()