# Trains an LSTM model with the same monthly walk-forward validation as LightGBM.
# The LSTM reads the last 7 days of hourly prices, and a small network combines that with
# the same features LightGBM uses. Summary goes to results/lstm_validation.csv.

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

# FEATURE_COLS and month_bounds come from features.py and evaluate.py, not from train_lgbm.py.
# Importing train_lgbm.py would also load LightGBM, and LightGBM and PyTorch bring different
# copies of the OpenMP threading library, which can crash or freeze Python on macOS.
from se3_electricity_forecast import tracking
from se3_electricity_forecast.evaluate import TEST_START, VALID_START, mae, month_bounds, rmse
from se3_electricity_forecast.features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "data" / "raw" / "prices_se3_hourly.parquet"
FEATURES = ROOT / "data" / "processed" / "features.parquet"
FORECAST_FEATURES = ROOT / "data" / "processed" / "features_forecast_weather.parquet"
PREDICTIONS = ROOT / "data" / "processed" / "lstm_valid_predictions.parquet"
RESULTS = ROOT / "results"

SEQ_LEN = 168          # hours of price history the LSTM reads (7 days x 24 hours)
HIDDEN = 32            # size of the LSTM's summary, kept small because the dataset is small
BATCH_SIZE = 256       # rows per training step
MAX_EPOCHS = 40        # maximum rounds through the training data
PATIENCE = 5           # stop after this many rounds without improvement
HOLDOUT_DAYS = 60      # days before each month used to decide when to stop training
LEARNING_RATE = 0.001  # standard step size for the Adam optimizer
SEED = 42              # makes the random parts repeatable

# A neural network would see hour 23 and hour 0 as far apart. Turning hour, weekday and
# month into sine and cosine puts them on a circle, where 23:00 and 00:00 are neighbours.
CYCLES = {"hour": 24, "weekday": 7, "month": 12}
INPUT_COLS = [c for c in FEATURE_COLS if c not in CYCLES] + [
    f"{col}_{f}" for col in CYCLES for f in ("sin", "cos")
]


class PriceLSTM(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        # Reads the price sequence, one price per hour
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, batch_first=True)
        # Combines the LSTM summary with the other features and outputs one price
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + n_inputs, 64),
            nn.ReLU(),        # lets the network learn non-linear relationships
            nn.Dropout(0.1),  # switches off 10% of connections during training to reduce overfitting
            nn.Linear(64, 1),
        )

    def forward(self, seq, tab):
        # h[-1] is the LSTM's final summary after reading all 168 hours
        _, (h, _) = self.lstm(seq)
        x = torch.cat([h[-1], tab], dim=1)
        return self.head(x).squeeze(1)


# Adds sine and cosine columns for hour, weekday and month.
def add_cyclical(df):
    df = df.copy()
    for col, period in CYCLES.items():
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df


# For each row, finds the position of 23:00 on the previous day in the price list.
# That is the last price really known in the morning, so the sequence ends there.
# It works with calendar dates instead of "minus 24 hours", so it stays correct
# on the days when the clock changes (23-hour and 25-hour days).
def add_sequence_end(df, prices):
    price_dates = pd.to_datetime(prices["time_local"].dt.date)
    last_index = pd.Series(np.arange(len(prices)), index=price_dates.to_numpy()).groupby(level=0).max()
    prev_dates = pd.to_datetime(df["time_local"].dt.date) - pd.Timedelta(days=1)
    df = df.copy()
    df["seq_end"] = last_index.loc[prev_dates.to_numpy()].to_numpy()
    return df


# Cuts out the 168 prices ending at each seq_end position, for all rows at once.
def make_sequences(price_values, seq_end):
    offsets = np.arange(-SEQ_LEN + 1, 1)
    return price_values[seq_end[:, None] + offsets]


# Scales the data and converts it to PyTorch tensors.
# Neural networks learn best when inputs are around 0 with a similar spread.
def to_tensors(part, price_values, stats):
    mu, sd, col_mean, col_std = stats
    seq = (make_sequences(price_values, part["seq_end"].to_numpy()) - mu) / sd
    tab = ((part[INPUT_COLS] - col_mean) / col_std).to_numpy()
    y = (part["price_sek_kwh"].to_numpy() - mu) / sd
    return (
        torch.tensor(seq, dtype=torch.float32).unsqueeze(-1),  # shape: rows x 168 hours x 1 value
        torch.tensor(tab, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )


# Trains one model with early stopping. Returns the best model, the scaling values
# it was trained with, and the number of epochs it ran.
def fit(train, holdout, price_values):
    # Scaling values come only from the training part. Using other parts would leak information.
    # replace(0, 1) avoids dividing by zero if a column never changes.
    stats = (
        train["price_sek_kwh"].mean(),
        train["price_sek_kwh"].std(),
        train[INPUT_COLS].mean(),
        train[INPUT_COLS].std().replace(0, 1),
    )
    seq_tr, tab_tr, y_tr = to_tensors(train, price_values, stats)
    seq_ho, tab_ho, y_ho = to_tensors(holdout, price_values, stats)

    torch.manual_seed(SEED)
    model = PriceLSTM(len(INPUT_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.L1Loss()  # absolute error, matches our main measure MAE

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        # Shuffle the rows every epoch, so the model does not learn their order
        order = torch.randperm(len(y_tr))
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(seq_tr[idx], tab_tr[idx]), y_tr[idx])
            loss.backward()  # works out how each weight should change
            # Limits the size of one update. Price spikes can otherwise make LSTM training unstable.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # updates the weights

        # Test on the holdout after every epoch and keep a copy of the best model so far
        model.eval()
        with torch.no_grad():
            holdout_loss = loss_fn(model(seq_ho, tab_ho), y_ho).item()
        if holdout_loss < best_loss:
            best_loss = holdout_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, stats, epoch


# Predicts prices (in SEK/kWh) for the rows in `predict` with a trained model.
def predict_prices(model, stats, predict, price_values):
    seq_pr, tab_pr, _ = to_tensors(predict, price_values, stats)
    model.eval()
    with torch.no_grad():
        pred = model(seq_pr, tab_pr).numpy()
    # Undo the scaling: back to SEK/kWh
    return pred * stats[1] + stats[0]


# Trains one model and predicts the given rows. Returns the predictions and the number
# of epochs, exactly as before the split into fit() and predict_prices().
def fit_predict(train, holdout, predict, price_values):
    model, stats, epoch = fit(train, holdout, price_values)
    return predict_prices(model, stats, predict, price_values), epoch


# Everything that defines the LSTM, saved with its MLflow run so it can be repeated exactly.
# Used for the validation run here and for the test run in final_test.py.
def mlflow_params(start, end):
    return {"seq_len": SEQ_LEN, "hidden": HIDDEN, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "holdout_days": HOLDOUT_DAYS, "learning_rate": LEARNING_RATE,
            "seed": SEED, "loss": "L1", "inputs": ",".join(INPUT_COLS), "period_start": str(start.date()),
            "period_end": str(end.date()), "retrain": "monthly walk-forward",
            "prediction_weather": "2-day-old forecasts", "device": "cpu", "torch_version": torch.__version__}


# Same idea as in train_lgbm.py: a new model for every month, trained only on earlier data.
# The 60 days just before each month are kept apart as a holdout for early stopping.
def walk_forward(train_df, predict_df, price_values, start, end):
    bounds = month_bounds(start, end)
    parts = []
    for month_start, month_end in zip(bounds[:-1], bounds[1:]):
        holdout_start = month_start - pd.Timedelta(days=HOLDOUT_DAYS)
        train = train_df[train_df["time_local"] < holdout_start]
        holdout = train_df[(train_df["time_local"] >= holdout_start) & (train_df["time_local"] < month_start)]
        predict = predict_df[(predict_df["time_local"] >= month_start) & (predict_df["time_local"] < month_end)]

        pred, epochs = fit_predict(train, holdout, predict, price_values)
        month_mae = mae(predict["price_sek_kwh"], pred)
        print(f"{month_start:%Y-%m}: {epochs} epochs, mae {month_mae:.4f}")

        parts.append(pd.DataFrame({
            "time_utc": predict["time_utc"].to_numpy(),
            "actual": predict["price_sek_kwh"].to_numpy(),
            "prediction": pred,
        }))
    return pd.concat(parts, ignore_index=True)


def main():
    prices = pd.read_parquet(PRICES).sort_values("time_utc").reset_index(drop=True)
    price_values = prices["price_sek_kwh"].to_numpy()
    # Train on measured weather, predict with the weather forecasts
    # unchanged in `features` for the MLflow data fingerprint, so it matches the other runs.
    features = pd.read_parquet(FEATURES)
    df = add_sequence_end(add_cyclical(features), prices)
    df_forecast = add_sequence_end(add_cyclical(pd.read_parquet(FORECAST_FEATURES)), prices)

    # Same benchmark as the baselines and LightGBM, so rel_mae is directly comparable
    baselines = pd.read_csv(RESULTS / "baselines.csv")
    naive_mae = baselines.query("split == 'valid' and model == 'weekly_naive'")["mae"].iloc[0]

    with tracking.start_run("lstm", stage="validation", model="lstm",
                            params=mlflow_params(VALID_START, TEST_START), data=features):
        preds = walk_forward(df, df_forecast, price_values, VALID_START, TEST_START)
        tracking.log_results(preds, naive_mae)
    preds.to_parquet(PREDICTIONS, index=False)

    lstm_mae = mae(preds["actual"], preds["prediction"])
    results = pd.DataFrame([{
        "model": "lstm",
        "hours": len(preds),
        "mae": lstm_mae,
        "rmse": rmse(preds["actual"], preds["prediction"]),
        "rel_mae": lstm_mae / naive_mae,
    }])
    results.to_csv(RESULTS / "lstm_validation.csv", index=False)

    # The LightGBM file is sorted by MAE, so the first row is the best version
    lgbm = pd.read_csv(RESULTS / "lgbm_validation.csv").iloc[0]
    print()
    print(results.round(4).to_string(index=False))
    print()
    print(f"best lightgbm: {lgbm['model']}, mae {lgbm['mae']:.4f}, rel_mae {lgbm['rel_mae']:.4f}")


if __name__ == "__main__":
    main()