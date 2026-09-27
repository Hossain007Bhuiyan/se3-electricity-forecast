from pathlib import Path
import lightgbm as lgb
import pandas as pd
from se3_electricity_forecast.evaluate import TEST_START, VALID_START, mae, rmse

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "features.parquet"
PREDICTIONS = ROOT / "data" / "processed" / "lgbm_valid_predictions.parquet"
RESULTS = ROOT / "results"

FEATURE_COLS = [
    "hour", "weekday", "month", "is_weekend", "is_holiday",
    "price_lag_24h", "price_lag_48h", "price_lag_168h",
    "prev_day_mean", "prev_day_min", "prev_day_max", "prev_day_std", "prev_7d_mean",
    "temperature_2m", "wind_speed_10m", "precipitation", "cloud_cover", "shortwave_radiation",
]

PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

CONFIGS = [
    {"name": "lgbm_price_l2", "target": "price", "objective": "l2"},
    {"name": "lgbm_price_l1", "target": "price", "objective": "l1"},
    {"name": "lgbm_diff_l2", "target": "diff", "objective": "l2"},
    {"name": "lgbm_diff_l1", "target": "diff", "objective": "l1"},
]


def make_target(df, target):
    if target == "diff":
        return df["price_sek_kwh"] - df["prev_day_mean"]
    return df["price_sek_kwh"]


def to_price(pred, df, target):
    if target == "diff":
        return pred + df["prev_day_mean"].to_numpy()
    return pred


def month_bounds(start, end):
    bounds = list(pd.date_range(start, end, freq="MS"))
    if bounds[-1] < end:
        bounds.append(end)
    return bounds


def walk_forward(df, config, start, end):
    bounds = month_bounds(start, end)
    parts = []
    for month_start, month_end in zip(bounds[:-1], bounds[1:]):
        train = df[df["time_local"] < month_start]
        test = df[(df["time_local"] >= month_start) & (df["time_local"] < month_end)]

        model = lgb.LGBMRegressor(objective=config["objective"], **PARAMS)
        model.fit(train[FEATURE_COLS], make_target(train, config["target"]))
        pred = to_price(model.predict(test[FEATURE_COLS]), test, config["target"])

        parts.append(pd.DataFrame({
            "time_utc": test["time_utc"].to_numpy(),
            "actual": test["price_sek_kwh"].to_numpy(),
            "prediction": pred,
        }))
    return pd.concat(parts, ignore_index=True)


def main():
    df = pd.read_parquet(FEATURES)
    baselines = pd.read_csv(RESULTS / "baselines.csv")
    valid_baselines = baselines[baselines["split"] == "valid"]
    naive_mae = valid_baselines.loc[valid_baselines["model"] == "weekly_naive", "mae"].iloc[0]
    best_baseline = valid_baselines.sort_values("mae").iloc[0]

    rows = []
    all_preds = []
    for config in CONFIGS:
        print("training", config["name"], "...")
        preds = walk_forward(df, config, VALID_START, TEST_START)
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

    results = pd.DataFrame(rows).sort_values("mae")
    results.to_csv(RESULTS / "lgbm_validation.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_parquet(PREDICTIONS, index=False)

    print()
    print(results.round(4).to_string(index=False))
    print()
    print(f"best baseline: {best_baseline['model']}, mae {best_baseline['mae']:.4f}, rel_mae {best_baseline['rel_mae']:.4f}")


if __name__ == "__main__":
    main()