# Experiment tracking with MLflow.
#
# Every training or evaluation run is saved as one MLflow "run": its settings (params), its
# results (metrics), where it came from (tags: stage, model, git commit, data fingerprint) and
# its hourly predictions (an artifact file). All runs can then be compared side by side in the
# MLflow web interface, and every result can be traced back to the exact code and data behind it.
#
# The runs are stored locally: metadata in mlflow.db (a SQLite database, MLflow's default
# backend) and files in the mlartifacts folder. Both are machine-specific, so they are not
# committed to Git.

import hashlib
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
import mlflow
import pandas as pd
from se3_electricity_forecast.evaluate import TZ, mae, rmse

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = ROOT / "mlartifacts"
EXPERIMENT = "se3-electricity-forecast"


# Returns the current Git commit and whether there are uncommitted changes, so every run
# can be traced back to the exact code. Returns "unknown" if Git is not available.
def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
        return commit, str(bool(changes))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


# A short fingerprint of a table's content. The same data always gives the same fingerprint,
# so two runs can be checked to have used exactly the same data.
def data_fingerprint(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()[:12]


# Starts an MLflow run and records what it is, where it came from and which data it used.
# Use it with "with", so the run is closed automatically, also if the script fails:
#     with tracking.start_run("lgbm_price_l1", stage="validation", model="lightgbm", params=..., data=df):
def start_run(run_name, stage, model, params, data):
    mlflow.set_tracking_uri(TRACKING_URI)
    # Create the experiment once, with its files stored in the project's mlartifacts folder
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(EXPERIMENT)

    run = mlflow.start_run(run_name=run_name)
    commit, uncommitted = git_state()
    mlflow.set_tags({
        "stage": stage,                      # "validation" or "test"
        "model": model,                      # "baseline", "lightgbm" or "lstm"
        "git_commit": commit,
        "uncommitted_changes": uncommitted,  # "True" means the code differed from the commit
    })
    mlflow.log_params(params)
    mlflow.log_params({
        "data_rows": len(data),
        "data_start": str(data["time_local"].min()),
        "data_end": str(data["time_local"].max()),
        "data_fingerprint": data_fingerprint(data),
        "python_version": platform.python_version(),
        "pandas_version": version("pandas"),
    })
    return run


# Logs the results of a finished run: MAE, RMSE and rel_mae over all predicted hours, the MAE
# of every month (as a chart in MLflow), and the hourly predictions as a file.
# preds needs the columns time_utc, actual and prediction.
def log_results(preds, naive_mae):
    model_mae = mae(preds["actual"], preds["prediction"])
    mlflow.log_metrics({
        "mae": model_mae,
        "rmse": rmse(preds["actual"], preds["prediction"]),
        "rel_mae": model_mae / naive_mae,
        "hours": len(preds),
    })

    # MAE per month, logged with step 1, 2, 3, ... so MLflow can draw it as a line
    month = pd.to_datetime(preds["time_utc"], utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m")
    errors = (preds["actual"] - preds["prediction"]).abs()
    monthly = errors.groupby(month.to_numpy()).mean()
    for step, value in enumerate(monthly.to_numpy(), start=1):
        mlflow.log_metric("monthly_mae", float(value), step=step)

    # The month names and the hourly predictions are saved as files next to the run
    with tempfile.TemporaryDirectory() as folder:
        monthly.rename_axis("month").rename("mae").to_csv(Path(folder) / "monthly_mae.csv")
        preds.to_parquet(Path(folder) / "predictions.parquet", index=False)
        mlflow.log_artifacts(folder)