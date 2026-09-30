# Trains LightGBM models with monthly walk-forward validation on the validation year.
# Four versions are compared (2 targets x 2 loss types). Summary goes to
# results/lgbm_validation.csv, all hourly predictions to data/processed/.

from pathlib import Path
import lightgbm as lgb
import pandas as pd
from se3_electricity_forecast import tracking
from se3_electricity_forecast.evaluate import TEST_START, VALID_START, mae, month_bounds, rmse
from se3_electricity_forecast.features import FEATURE_COLS


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "features.parquet"
FORECAST_FEATURES = ROOT / "data" / "processed" / "features_forecast_weather.parquet"
PREDICTIONS = ROOT / "data" / "processed" / "lgbm_valid_predictions.parquet"
RESULTS = ROOT / "results"

# Fixed, standard settings. They are not tuned, to keep the validation result honest.
PARAMS = {
    "n_estimators": 600,         # number of small trees
    "learning_rate": 0.03,       # each tree only corrects a little, which is more stable
    "num_leaves": 31,            # limits how detailed each tree can be
    "min_child_samples": 50,     # each leaf needs at least 50 rows, which reduces overfitting
    "subsample": 0.8,            # each tree sees a random 80% of the rows...
    "subsample_freq": 1,         # ...drawn again for every tree
    "colsample_bytree": 0.8,     # and a random 80% of the columns
    "random_state": 42,          # makes the randomness repeatable
    "verbose": -1,               # hides LightGBM's internal messages
}

# target "price": predict the price directly.
# target "diff": predict the difference from yesterday's average, then add it back.
# objective "l2" focuses on squared errors (big misses), "l1" on absolute errors (matches MAE).
CONFIGS = [
    {"name": "lgbm_price_l2", "target": "price", "objective": "l2"},
    {"name": "lgbm_price_l1", "target": "price", "objective": "l1"},
    {"name": "lgbm_diff_l2", "target": "diff", "objective": "l2"},
    {"name": "lgbm_diff_l1", "target": "diff", "objective": "l1"},
]


# Returns what the model should learn, depending on the chosen target.
def make_target(df, target):
    if target == "diff":
        return df["price_sek_kwh"] - df["prev_day_mean"]
    return df["price_sek_kwh"]


# Turns the model output back into a price in SEK/kWh.
def to_price(pred, df, target):
    if target == "diff":
        return pred + df["prev_day_mean"].to_numpy()
    return pred


# Everything that defines one LightGBM version, saved with its MLflow run so it can be
# repeated exactly. Used for the validation runs here and for the test run in final_test.py.
def mlflow_params(config, start, end):
    return {**PARAMS, "target": config["target"], "objective": config["objective"],
            "features": ",".join(FEATURE_COLS), "period_start": str(start.date()),
            "period_end": str(end.date()), "retrain": "monthly walk-forward",
            "prediction_weather": "2-day-old forecasts", "lightgbm_version": lgb.__version__}


# For each month: train a new model on everything before that month, then predict the month.
# train_df has measured weather, predict_df has the weather forecasts.
def walk_forward(train_df, predict_df, config, start, end):
    bounds = month_bounds(start, end)
    parts = []
    for month_start, month_end in zip(bounds[:-1], bounds[1:]):
        train = train_df[train_df["time_local"] < month_start]
        test = predict_df[(predict_df["time_local"] >= month_start) & (predict_df["time_local"] < month_end)]

        model = lgb.LGBMRegressor(objective=config["objective"], **PARAMS)
        model.fit(train[FEATURE_COLS], make_target(train, config["target"]))
        pred = to_price(model.predict(test[FEATURE_COLS]), test, config["target"])

        # .to_numpy() places values by position, not by pandas row labels
        parts.append(pd.DataFrame({
            "time_utc": test["time_utc"].to_numpy(),
            "actual": test["price_sek_kwh"].to_numpy(),
            "prediction": pred,
        }))
    return pd.concat(parts, ignore_index=True)


def main():
    df = pd.read_parquet(FEATURES)
    df_forecast = pd.read_parquet(FORECAST_FEATURES)

    # Use the same benchmark as in baselines.py, so rel_mae is directly comparable
    baselines = pd.read_csv(RESULTS / "baselines.csv")
    valid_baselines = baselines[baselines["split"] == "valid"]
    naive_mae = valid_baselines.loc[valid_baselines["model"] == "weekly_naive", "mae"].iloc[0]
    best_baseline = valid_baselines.sort_values("mae").iloc[0]

    rows = []
    all_preds = []
    for config in CONFIGS:
        print("training", config["name"], "...")
        params = mlflow_params(config, VALID_START, TEST_START)
        with tracking.start_run(config["name"], stage="validation", model="lightgbm", params=params, data=df):
            preds = walk_forward(df, df_forecast, config, VALID_START, TEST_START)
            tracking.log_results(preds, naive_mae)
        preds["model"] = config["name"]
        all_preds.append(preds)
        model_mae = mae(preds["actual"], preds["prediction"])
        rows.append({
            "model": config["name"],
            "hours": len(preds),
            "mae": model_mae,
            "rmse": rmse(preds["actual"], preds["prediction"]),
            "rel_mae": model_mae / naive_mae,
        })

    # Sorted by MAE, so the best version is the first row
    results = pd.DataFrame(rows).sort_values("mae")
    results.to_csv(RESULTS / "lgbm_validation.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_parquet(PREDICTIONS, index=False)

    print()
    print(results.round(4).to_string(index=False))
    print()
    print(f"best baseline: {best_baseline['model']}, mae {best_baseline['mae']:.4f}, rel_mae {best_baseline['rel_mae']:.4f}")


if __name__ == "__main__":
    main()