# Explains what drives the predictions of the two final models.
#
# Both models are trained once on all data before the test year, like the models that would
# have been used on 1 Oct 2025, and are explained on the test year with weather forecasts as
# input. This is analysis only: no model decisions are made from it.
#
# The work is split into three commands that run one after the other:
#   python -m se3_electricity_forecast.explain lgbm    -> SHAP values and permutation importance for LightGBM
#   python -m se3_electricity_forecast.explain lstm    -> SHAP values (expected gradients) and permutation importance for the LSTM
#   python -m se3_electricity_forecast.explain report  -> compares both models in one chart
# LightGBM and PyTorch are never loaded in the same process (see final_test.py for why), so
# train_lgbm and train_lstm are imported inside the function that needs them.

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts straight to files, without opening windows
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from se3_electricity_forecast.evaluate import TEST_END, TEST_START, mae
from se3_electricity_forecast.features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "data" / "raw" / "prices_se3_hourly.parquet"
FEATURES = ROOT / "data" / "processed" / "features.parquet"
FORECAST_FEATURES = ROOT / "data" / "processed" / "features_forecast_weather.parquet"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LGBM_NAME = "lgbm_price_l1"  # the LightGBM version chosen on the validation year
REPEATS = 3                  # each input is shuffled 3 times and the results averaged
SEED = 42                    # makes the shuffling repeatable

# Expected-gradients settings for the LSTM (see lstm_expected_gradients below)
SHAP_REFERENCES = 50         # random training hours used as reference ("average") inputs
SHAP_STEPS = 16              # points on the path from each reference hour to the explained hour
SHAP_MAX_SEQUENCES = 2048    # max sequences the LSTM handles at once (hours x steps), limits memory


# Returns the test-year rows (8664 hours), with a clean 0..n-1 index.
def test_rows(df):
    return df[(df["time_local"] >= TEST_START) & (df["time_local"] < TEST_END)].reset_index(drop=True)


# Permutation importance: shuffle one input (or a group of inputs) across the test rows and
# measure how much the MAE gets worse. The bigger the increase, the more the model relies on it.
# Works the same way for any model, which makes LightGBM and the LSTM directly comparable.
def permutation_importance(predict_fn, data, groups):
    rng = np.random.default_rng(SEED)
    base_mae = mae(data["price_sek_kwh"], predict_fn(data))
    rows = []
    for name, cols in groups.items():
        increases = []
        for _ in range(REPEATS):
            shuffled = data.copy()
            order = rng.permutation(len(data))
            # All columns of a group get the same shuffle, so they stay consistent with each other
            shuffled[cols] = data[cols].to_numpy()[order]
            increases.append(mae(data["price_sek_kwh"], predict_fn(shuffled)) - base_mae)
        rows.append({"feature": name, "mae_increase": float(np.mean(increases))})
    return base_mae, pd.DataFrame(rows).sort_values("mae_increase", ascending=False)


# The two colours of the shap library: blue for low values / negative impact, red for
# high values / positive impact. The colour map runs from blue through purple to red.
SHAP_BLUE = "#008bfb"
SHAP_RED = "#ff0051"
SHAP_CMAP = LinearSegmentedColormap.from_list("shap_red_blue", [SHAP_BLUE, SHAP_RED])


# Colour range for one feature, with the same rule as the shap library: the 5th to 95th
# percentile, so a few extreme values do not squeeze all other dots into one colour. If those
# are equal (e.g. is_holiday is 0 in far more than 95% of hours), the 1st to 99th percentile
# is tried, and finally the min and max.
def colour_range(x):
    low, high = np.percentile(x, [5, 95])
    if low == high:
        low, high = np.percentile(x, [1, 99])
        if low == high:
            low, high = x.min(), x.max()
    return low, high


# Vertical offsets for the dots of one beeswarm row, with the same method as the shap library:
# the x-axis is cut into 100 bins, and the dots in each bin are stacked in layers, alternating
# above and below the row. Where many hours have similar SHAP values, the row gets wider,
# so the shape of each row shows where most of the dots are.
def swarm_offsets(values, rng, row_height=0.4):
    bins = np.round(100 * (values - values.min()) / (values.max() - values.min() + 1e-8))
    order = np.argsort(bins + rng.normal(size=len(values)) * 1e-6)  # tiny noise breaks ties randomly
    offsets = np.zeros(len(values))
    layer = 0
    last_bin = -1
    for i in order:
        if bins[i] != last_bin:
            layer = 0
        offsets[i] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)  # 0, +1, -1, +2, -2, ...
        layer += 1
        last_bin = bins[i]
    return offsets * 0.9 * (row_height / np.max(offsets + 1))


# SHAP summary (beeswarm) chart, drawn like shap.plots.beeswarm: one row per feature, one dot
# per test hour. The dot's position shows how much that feature pushed that hour's forecast
# up (right) or down (left), and its colour shows whether the feature's value was low (blue)
# or high (red). Features are sorted by mean |SHAP|, the most important at the top.
# All features are shown (the shap library shows 10 by default, but can show more).
# shap_values and feature_values must have the same columns in the same order.
def plot_shap_beeswarm(shap_values, feature_values, title, filename):
    rng = np.random.default_rng(SEED)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_values.columns)
    order = mean_abs_shap.sort_values().index  # least important first, so it ends up at the bottom
    fig, ax = plt.subplots(figsize=(9, 8))
    for row, feature in enumerate(order):
        col = feature_values.columns.get_loc(feature)
        x = feature_values[feature].to_numpy(dtype=float)
        low, high = colour_range(x)
        # Colour value between 0 (low) and 1 (high), so all rows share one colour bar
        colour = np.clip((x - low) / (high - low), 0, 1) if high > low else np.full(len(x), 0.5)
        ax.axhline(row, color="#cccccc", linewidth=0.5, dashes=(1, 5), zorder=-1)
        points = ax.scatter(shap_values[:, col], row + swarm_offsets(shap_values[:, col], rng), c=colour,
                            cmap=SHAP_CMAP, vmin=0, vmax=1, s=12, linewidth=0, rasterized=True)
    ax.axvline(0, color="#999999", linewidth=0.8, zorder=-1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-1, len(order))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("SHAP value (impact on the forecast, SEK/kWh)")
    ax.set_title(title)
    bar = fig.colorbar(points, ax=ax, ticks=[0, 1], aspect=50)
    bar.ax.set_yticklabels(["Low", "High"])
    bar.set_label("Feature value", labelpad=-10)
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


# SHAP waterfall chart for one single hour, laid out like shap.plots.waterfall. It is read
# from the bottom up: it starts at E[f(X)], the model's average forecast, and each row adds one
# feature's contribution (red pushes up, blue pushes down) until it reaches f(x) at the top.
# The 9 biggest contributions get their own row (biggest at the top), and the remaining
# features are grouped in the bottom row. Each label shows the feature's value for this hour,
# e.g. "-8.500 = temperature_2m".
# contributions and feature_values are pandas Series with the feature names as index.
def plot_shap_waterfall(contributions, base, feature_values, title, filename, max_display=10):
    forecast = base + contributions.sum()
    order = contributions.abs().sort_values(ascending=False).index
    shown = order[: max_display - 1]
    rest = order[max_display - 1:]

    # Start at the forecast at the top and walk down, removing one contribution per row
    rows = []
    loc = forecast
    for name in shown:
        loc -= contributions[name]
        rows.append((f"{feature_values[name]:.3f} = {name}", contributions[name], loc))
    rows.append((f"{len(rest)} other features", contributions[rest].sum(), base))

    positions = np.array([base, forecast] + [left for _, _, left in rows] + [left + v for _, v, left in rows])
    pad = max((positions.max() - positions.min()) * 0.1, 0.01)
    span = positions.max() - positions.min() + 2 * pad

    fig, ax = plt.subplots(figsize=(9, 6))
    for y, (label, value, left) in zip(range(len(rows) - 1, -1, -1), rows):
        colour = SHAP_RED if value > 0 else SHAP_BLUE
        ax.barh(y, value, left=left, height=0.6, color=colour)
        # Write the value inside the bar if it fits, otherwise just outside it
        if abs(value) > 0.08 * span:
            ax.text(left + value / 2, y, f"{value:+.3f}", ha="center", va="center", color="white", fontsize=8)
        elif value > 0:
            ax.text(left + value, y, f" {value:+.3f}", ha="left", va="center", color=colour, fontsize=8)
        else:
            ax.text(left + value, y, f"{value:+.3f} ", ha="right", va="center", color=colour, fontsize=8)
    ax.set_yticks(range(len(rows) - 1, -1, -1))
    ax.set_yticklabels([label for label, _, _ in rows])
    ax.set_xlim(positions.min() - pad, positions.max() + pad)

    # Dashed line at E[f(X)] (bottom label), solid line at f(x) (top label)
    ax.axvline(base, color="#999999", linestyle="--", linewidth=0.8)
    ax.axvline(forecast, color="#333333", linewidth=0.8)
    ax.annotate(f"f(x) = {forecast:.3f}", xy=(forecast, 1), xycoords=("data", "axes fraction"),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
    ax.annotate(f"E[f(X)] = {base:.3f}", xy=(base, 0), xycoords=("data", "axes fraction"),
                xytext=(0, -22), textcoords="offset points", ha="center", va="top")
    ax.set_xlabel("SEK/kWh", labelpad=24)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, pad=28)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


# Title for a waterfall chart: which model, which hour, and the real price in that hour.
def waterfall_title(model_name, row):
    time = row["time_local"].strftime("%Y-%m-%d %H:%M")
    return f"{model_name} forecast for {time} (actual price {row['price_sek_kwh']:.2f} SEK/kWh)"


# SHAP values for the LSTM with expected gradients, the method behind GradientSHAP, computed
# with Captum (PyTorch's explainability library). Exact SHAP values are not possible for a
# neural network. Instead, for each reference hour, Integrated Gradients follows the straight
# path from that reference hour to the explained hour and splits the change in the forecast
# between the inputs. Averaging over all reference hours gives each input's contribution
# relative to E[f(X)], the average forecast of the reference hours.
# GradientSHAP estimates the same thing with random samples. Here every reference hour and a
# fixed set of points on each path are used instead, so the result is the same on every run,
# and base + contributions equals the forecast except for a small error from using a limited
# number of points on each path.
# The contributions of the 168 price-history values are added up into one "price_history_168h"
# value, and each sine/cosine pair into hour, weekday or month, so the result has one column
# per feature like the LightGBM SHAP values. Returns the contributions in SEK/kWh.
def lstm_expected_gradients(train_lstm, model, stats, references, data, price_values):
    from captum.attr import IntegratedGradients

    seq_ref, tab_ref, _ = train_lstm.to_tensors(references, price_values, stats)
    seq, tab, _ = train_lstm.to_tensors(data, price_values, stats)
    explainer = IntegratedGradients(model)

    # Captum runs every hour SHAP_STEPS times, so this many hours fit in one batch
    batch = max(1, SHAP_MAX_SEQUENCES // SHAP_STEPS)
    seq_parts, tab_parts = [], []
    for i in range(0, len(data), batch):
        seq_batch, tab_batch = seq[i:i + batch], tab[i:i + batch]
        seq_sum = np.zeros(seq_batch.shape[:2])
        tab_sum = np.zeros(tab_batch.shape)
        for k in range(len(references)):
            # Use reference hour k as the starting point for every hour in the batch
            baselines = (seq_ref[k:k + 1].expand_as(seq_batch), tab_ref[k:k + 1].expand_as(tab_batch))
            a_seq, a_tab = explainer.attribute((seq_batch, tab_batch), baselines=baselines, n_steps=SHAP_STEPS,
                                               method="gausslegendre", internal_batch_size=SHAP_MAX_SEQUENCES)
            seq_sum += a_seq.detach().numpy()[:, :, 0]
            tab_sum += a_tab.detach().numpy()
        seq_parts.append(seq_sum / len(references))
        tab_parts.append(tab_sum / len(references))
    a_seq = np.concatenate(seq_parts)
    a_tab = np.concatenate(tab_parts)

    # The model works on scaled prices, so multiply by the price spread to get SEK/kWh
    sd = stats[1]
    columns = {}
    for col in FEATURE_COLS:
        if col in train_lstm.CYCLES:
            idx = [train_lstm.INPUT_COLS.index(f"{col}_sin"), train_lstm.INPUT_COLS.index(f"{col}_cos")]
        else:
            idx = [train_lstm.INPUT_COLS.index(col)]
        columns[col] = a_tab[:, idx].sum(axis=1) * sd
    columns["price_history_168h"] = a_seq.sum(axis=1) * sd
    return pd.DataFrame(columns, index=data.index)


def run_lgbm():
    from se3_electricity_forecast import train_lgbm  # loads LightGBM only, never PyTorch

    # Train on measured weather before the test year, explain with weather forecasts
    df = pd.read_parquet(FEATURES)
    test = test_rows(pd.read_parquet(FORECAST_FEATURES))
    train = df[df["time_local"] < TEST_START]

    print("training LightGBM ...")
    config = next(c for c in train_lgbm.CONFIGS if c["name"] == LGBM_NAME)
    model = train_lgbm.lgb.LGBMRegressor(objective=config["objective"], **train_lgbm.PARAMS)
    model.fit(train[FEATURE_COLS], train_lgbm.make_target(train, config["target"]))

    def predict_fn(data):
        return train_lgbm.to_price(model.predict(data[FEATURE_COLS]), data, config["target"])

    # SHAP values: how much each feature pushed each single prediction up or down, in SEK/kWh.
    # pred_contrib=True makes LightGBM compute exact TreeSHAP values itself, so no extra package
    # is needed. The last column is the base value that all contributions start from.
    contrib = model.predict(test[FEATURE_COLS], pred_contrib=True)
    shap_values = contrib[:, :-1]
    # Check: base value + all contributions must add up to the normal prediction for every hour
    print("SHAP values add up to the predictions:", np.allclose(contrib.sum(axis=1), predict_fn(test)))
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLS)

    print("computing permutation importance ...")
    groups = {col: [col] for col in FEATURE_COLS}
    base_mae, importance = permutation_importance(predict_fn, test, groups)
    importance["mean_abs_shap"] = importance["feature"].map(mean_abs_shap)
    importance.to_csv(RESULTS / "importance_lgbm.csv", index=False)

    plot_shap_beeswarm(shap_values, test[FEATURE_COLS], "LightGBM: SHAP summary on the test year",
                       "shap_beeswarm_lgbm.png")

    # Waterfall for the single most expensive hour of the test year
    spike = int(test["price_sek_kwh"].idxmax())
    plot_shap_waterfall(pd.Series(shap_values[spike], index=FEATURE_COLS), contrib[spike, -1],
                        test.loc[spike, FEATURE_COLS], waterfall_title("LightGBM", test.loc[spike]),
                        "shap_waterfall_lgbm.png")

    print(f"test MAE of this single model: {base_mae:.4f}")
    print(importance.round(4).to_string(index=False))


def run_lstm():
    from se3_electricity_forecast import train_lstm  # loads PyTorch only, never LightGBM

    df = pd.read_parquet(FEATURES)
    df_forecast = pd.read_parquet(FORECAST_FEATURES)
    prices = pd.read_parquet(PRICES).sort_values("time_utc").reset_index(drop=True)
    price_values = prices["price_sek_kwh"].to_numpy()

    # Same preparation as in train_lstm.py: sine/cosine columns and the 7-day sequence position
    df_lstm = train_lstm.add_sequence_end(train_lstm.add_cyclical(df), prices)
    test = test_rows(train_lstm.add_sequence_end(train_lstm.add_cyclical(df_forecast), prices))

    # Same split as in train_lstm.py: the 60 days before the test year are the holdout
    holdout_start = TEST_START - pd.Timedelta(days=train_lstm.HOLDOUT_DAYS)
    train = df_lstm[df_lstm["time_local"] < holdout_start]
    holdout = df_lstm[(df_lstm["time_local"] >= holdout_start) & (df_lstm["time_local"] < TEST_START)]

    print("training LSTM ...")
    model, stats, epochs = train_lstm.fit(train, holdout, price_values)
    print("trained for", epochs, "epochs")

    def predict_fn(data):
        return train_lstm.predict_prices(model, stats, data, price_values)

    # Shuffle the same 18 inputs as for LightGBM. Hour, weekday and month are stored as
    # sine and cosine pairs in the LSTM, so each pair is shuffled together. The price history
    # is shuffled by giving each hour the 168-hour sequence of another, random hour.
    groups = {col: [col] for col in FEATURE_COLS if col not in train_lstm.CYCLES}
    groups.update({col: [f"{col}_sin", f"{col}_cos"] for col in train_lstm.CYCLES})
    groups["price_history_168h"] = ["seq_end"]

    print("computing permutation importance ...")
    base_mae, importance = permutation_importance(predict_fn, test, groups)

    # Reference hours: a random sample of the training hours.
    # E[f(X)], the starting point of every explanation, is the average forecast for them.
    rng = np.random.default_rng(SEED)
    references = train.iloc[rng.choice(len(train), SHAP_REFERENCES, replace=False)]
    base = float(predict_fn(references).mean())

    print("computing SHAP values with expected gradients (this takes a while) ...")
    shap_values = lstm_expected_gradients(train_lstm, model, stats, references, test, price_values)

    # Check how close the result is: base + contributions should equal the forecast
    forecasts = predict_fn(test)
    gap = np.abs(base + shap_values.sum(axis=1).to_numpy() - forecasts)
    distance = np.abs(forecasts - base)
    print(f"SHAP check: base + contributions differ from the forecast by {gap.mean():.4f} SEK/kWh on average "
          f"({gap.mean() / distance.mean():.1%} of the average distance between forecast and base)")

    importance["mean_abs_shap"] = importance["feature"].map(shap_values.abs().mean())
    importance.to_csv(RESULTS / "importance_lstm.csv", index=False)

    # Feature values for the chart colours. For the price history, the value shown is the
    # average price of those 168 hours.
    feature_values = test[FEATURE_COLS].copy()
    feature_values["price_history_168h"] = train_lstm.make_sequences(price_values, test["seq_end"].to_numpy()).mean(axis=1)
    plot_shap_beeswarm(shap_values.to_numpy(), feature_values, "LSTM: SHAP summary on the test year (expected gradients)",
                       "shap_beeswarm_lstm.png")

    # Waterfall for the same most expensive hour as for LightGBM
    spike = int(test["price_sek_kwh"].idxmax())
    spike_values = shap_values.loc[spike]
    print(f"waterfall hour: model forecast {forecasts[spike]:.3f}, base + contributions {base + spike_values.sum():.3f}")
    # The approximation means f(x) in the chart (base + contributions) can differ a little from
    # the model's real forecast. The title shows both, so nothing is hidden.
    title = (waterfall_title("LSTM", test.loc[spike])
             + f"\nExpected gradients (SHAP approximation), real model forecast {forecasts[spike]:.3f}")
    plot_shap_waterfall(spike_values, base, feature_values.loc[spike], title, "shap_waterfall_lstm.png")

    print(f"test MAE of this single model: {base_mae:.4f}")
    print(importance.round(4).to_string(index=False))


def report():
    lgbm = pd.read_csv(RESULTS / "importance_lgbm.csv")
    lstm = pd.read_csv(RESULTS / "importance_lstm.csv")

    # One row per input, one column per model. The 168-hour price history only exists in
    # the LSTM, so LightGBM has no value there.
    both = lgbm[["feature", "mae_increase"]].merge(
        lstm[["feature", "mae_increase"]], on="feature", how="outer", suffixes=("_lgbm", "_lstm")
    ).sort_values("mae_increase_lstm")

    # One panel per model, each sorted by its own importance. Both panels use the same
    # x-axis scale, so the bar lengths can be compared between the models.
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True)
    for ax, (name, table, colour) in zip(axes, [("LightGBM", lgbm, "tab:blue"), ("LSTM", lstm, "tab:orange")]):
        table = table.sort_values("mae_increase")
        ax.barh(table["feature"], table["mae_increase"], color=colour)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("MAE increase when shuffled (SEK/kWh)")
    fig.suptitle("Permutation importance on the test year (price_history_168h is an LSTM-only input)")
    fig.tight_layout()
    fig.savefig(FIGURES / "importance_comparison.png", dpi=150)
    plt.close(fig)

    print(both.sort_values("mae_increase_lstm", ascending=False).round(4).to_string(index=False))


def main():
    # The part to run is given after the module name, e.g. "... explain lgbm"
    parts = {"lgbm": run_lgbm, "lstm": run_lstm, "report": report}
    if len(sys.argv) != 2 or sys.argv[1] not in parts:
        print("usage: python -m se3_electricity_forecast.explain lgbm|lstm|report")
        return
    parts[sys.argv[1]]()


if __name__ == "__main__":
    main()