# Final evaluation on the test year (1 Oct 2025 - 26 Sep 2026).
# Runs the chosen LightGBM version and the LSTM with the same monthly walk-forward setup
# as on the validation year, and compares them with the baselines on exactly the same hours.
# The test year is used only once, after all model choices were made on the validation year.
#
# The work is split into three commands that run one after the other:
#   python -m se3_electricity_forecast.final_test lgbm    -> trains LightGBM, saves its predictions
#   python -m se3_electricity_forecast.final_test lstm    -> trains the LSTM, saves its predictions
#   python -m se3_electricity_forecast.final_test report  -> compares everything and saves the results
#
# LightGBM and PyTorch bring different copies of the OpenMP threading library. Loading both
# into the same Python process can freeze or crash Python on macOS. That is why each model
# runs in its own command, and why train_lgbm and train_lstm are imported inside the function
# that needs them instead of at the top of this file: a process only ever loads one of them.

import sys
from pathlib import Path

import pandas as pd

from se3_electricity_forecast.baselines import baseline_predictions
from se3_electricity_forecast.evaluate import TEST_END, TEST_START, mae, rmse, split

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "data" / "raw" / "prices_se3_hourly.parquet"
FEATURES = ROOT / "data" / "processed" / "features.parquet"
FORECAST_FEATURES = ROOT / "data" / "processed" / "features_forecast_weather.parquet"
LGBM_PREDICTIONS = ROOT / "data" / "processed" / "test_predictions_lgbm.parquet"
LSTM_PREDICTIONS = ROOT / "data" / "processed" / "test_predictions_lstm.parquet"
RESULTS = ROOT / "results"

# The LightGBM version that was best on the validation year
LGBM_NAME = "lgbm_price_l1"


# Trains LightGBM month by month on the test year and saves the hourly predictions.
def run_lgbm():
    from se3_electricity_forecast import train_lgbm  # loads LightGBM only, never PyTorch

    # Measured weather for training, weather forecasts for predicting, same as before
    df = pd.read_parquet(FEATURES)
    df_forecast = pd.read_parquet(FORECAST_FEATURES)

    print("training LightGBM ...")
    lgbm_config = next(c for c in train_lgbm.CONFIGS if c["name"] == LGBM_NAME)
    preds = train_lgbm.walk_forward(df, df_forecast, lgbm_config, TEST_START, TEST_END)
    preds.to_parquet(LGBM_PREDICTIONS, index=False)
    print("saved", len(preds), "LightGBM predictions")


# Trains the LSTM month by month on the test year and saves the hourly predictions.
def run_lstm():
    from se3_electricity_forecast import train_lstm  # loads PyTorch only, never LightGBM

    df = pd.read_parquet(FEATURES)
    df_forecast = pd.read_parquet(FORECAST_FEATURES)
    prices = pd.read_parquet(PRICES).sort_values("time_utc").reset_index(drop=True)
    price_values = prices["price_sek_kwh"].to_numpy()

    # Same preparation as in train_lstm.py: sine/cosine columns and the 7-day sequence position
    df_lstm = train_lstm.add_sequence_end(train_lstm.add_cyclical(df), prices)
    df_forecast_lstm = train_lstm.add_sequence_end(train_lstm.add_cyclical(df_forecast), prices)

    print("training LSTM (one line per finished month, the first one can take a while) ...")
    preds = train_lstm.walk_forward(df_lstm, df_forecast_lstm, price_values, TEST_START, TEST_END)
    preds.to_parquet(LSTM_PREDICTIONS, index=False)
    print("saved", len(preds), "LSTM predictions")


# Reads both prediction files, checks them, and compares all models on the test hours.
def report():
    df = pd.read_parquet(FEATURES)
    lgbm_preds = pd.read_parquet(LGBM_PREDICTIONS)
    lstm_preds = pd.read_parquet(LSTM_PREDICTIONS)

    # The test hours, exactly as defined in evaluate.py (8664 hours)
    _, _, test = split(df)

    # Both models must have predicted exactly the same hours as the baselines,
    # otherwise the comparison would not be fair
    test_hours = set(test["time_utc"])
    print("LightGBM predicted the same hours as the baselines:", set(lgbm_preds["time_utc"]) == test_hours)
    print("LSTM predicted the same hours as the baselines:", set(lstm_preds["time_utc"]) == test_hours)

    # One row per model: the four baselines, then the two trained models
    rows = []
    for name, pred in baseline_predictions(test).items():
        rows.append({"model": name, "hours": len(test), "mae": mae(test["price_sek_kwh"], pred),
                     "rmse": rmse(test["price_sek_kwh"], pred)})
    for name, preds in [(LGBM_NAME, lgbm_preds), ("lstm", lstm_preds)]:
        rows.append({"model": name, "hours": len(preds), "mae": mae(preds["actual"], preds["prediction"]),
                     "rmse": rmse(preds["actual"], preds["prediction"])})

    # rel_mae against weekly_naive on the test year, the same benchmark as in the other steps
    results = pd.DataFrame(rows)
    naive_mae = results.loc[results["model"] == "weekly_naive", "mae"].iloc[0]
    results["rel_mae"] = results["mae"] / naive_mae
    results = results.sort_values("mae")
    results.to_csv(RESULTS / "test_results.csv", index=False)

    print()
    print(results.round(4).to_string(index=False))


def main():
    # The part to run is given after the module name, e.g. "... final_test lgbm"
    parts = {"lgbm": run_lgbm, "lstm": run_lstm, "report": report}
    if len(sys.argv) != 2 or sys.argv[1] not in parts:
        print("usage: python -m se3_electricity_forecast.final_test lgbm|lstm|report")
        return
    parts[sys.argv[1]]()


if __name__ == "__main__":
    main()