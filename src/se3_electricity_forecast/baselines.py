from pathlib import Path

import numpy as np
import pandas as pd

from se3_electricity_forecast.evaluate import mae, rmse, split

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "features.parquet"
RESULTS = ROOT / "results"


def baseline_predictions(df):
    use_last_week = df["weekday"].isin([0, 5, 6])
    return {
        "same_hour_yesterday": df["price_lag_24h"],
        "same_hour_last_week": df["price_lag_168h"],
        "weekly_naive": np.where(use_last_week, df["price_lag_168h"], df["price_lag_24h"]),
        "yesterday_mean": df["prev_day_mean"],
    }


def main():
    df = pd.read_parquet(FEATURES)
    train, valid, test = split(df)

    for name, part in [("train", train), ("valid", valid), ("test", test)]:
        print(f"{name}: {len(part)} rows, {part['time_local'].min()} to {part['time_local'].max()}")

    rows = []
    for split_name, part in [("valid", valid), ("test", test)]:
        for model, pred in baseline_predictions(part).items():
            rows.append({
                "split": split_name,
                "model": model,
                "mae": mae(part["price_sek_kwh"], pred),
                "rmse": rmse(part["price_sek_kwh"], pred),
            })

    results = pd.DataFrame(rows)
    naive_mae = results[results["model"] == "weekly_naive"].set_index("split")["mae"]
    results["rel_mae"] = results["mae"] / results["split"].map(naive_mae)

    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "baselines.csv", index=False)

    for split_name in ["valid", "test"]:
        print()
        print(split_name)
        print(results[results["split"] == split_name].drop(columns="split").round(4).to_string(index=False))


if __name__ == "__main__":
    main()