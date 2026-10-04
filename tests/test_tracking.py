# Tests for tracking.py. MLflow writes to a temporary folder here, never to the project's mlflow.db.

import mlflow
import pandas as pd
from se3_electricity_forecast import tracking


def test_data_fingerprint_is_stable_and_detects_changes(prices):
    assert tracking.data_fingerprint(prices) == tracking.data_fingerprint(prices.copy())
    changed = prices.copy()
    changed.loc[5, "price_sek_kwh"] += 0.001
    assert tracking.data_fingerprint(changed) != tracking.data_fingerprint(prices)


def test_start_run_and_log_results(prices, tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(tracking, "ARTIFACT_ROOT", tmp_path / "mlartifacts")
    preds = pd.DataFrame({
        "time_utc": prices["time_utc"],
        "actual": prices["price_sek_kwh"],
        "prediction": prices["price_sek_kwh"] + 0.5,  # always 0.5 too high
    })
    with tracking.start_run("test_run", stage="validation", model="baseline",
                            params={"rule": "test"}, data=prices) as run:
        tracking.log_results(preds, naive_mae=1.0)

    client = mlflow.MlflowClient()
    logged = client.get_run(run.info.run_id)
    assert abs(logged.data.metrics["mae"] - 0.5) < 1e-12
    assert abs(logged.data.metrics["rel_mae"] - 0.5) < 1e-12
    assert logged.data.metrics["hours"] == len(preds)
    assert logged.data.tags["stage"] == "validation"
    assert logged.data.params["data_fingerprint"] == tracking.data_fingerprint(prices)
    # The test period covers October and November 2025, so two monthly values
    assert len(client.get_metric_history(run.info.run_id, "monthly_mae")) == 2
    files = sorted(a.path for a in client.list_artifacts(run.info.run_id))
    assert files == ["monthly_mae.csv", "predictions.parquet"]


def test_export_runs_writes_runs_and_monthly_mae(prices, tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(tracking, "ARTIFACT_ROOT", tmp_path / "mlartifacts")
    preds = pd.DataFrame({"time_utc": prices["time_utc"], "actual": prices["price_sek_kwh"],
                          "prediction": prices["price_sek_kwh"] + 0.5})
    for name, stage in [("model_a", "validation"), ("model_a", "test")]:
        with tracking.start_run(name, stage=stage, model="baseline", params={"rule": "test"}, data=prices):
            tracking.log_results(preds, naive_mae=1.0)
    deleted = tracking.start_run("deleted_run", stage="validation", model="baseline", params={}, data=prices)
    tracking.log_results(preds, naive_mae=1.0)
    mlflow.end_run()
    mlflow.delete_run(deleted.info.run_id)  # deleted runs must not be exported

    tracking.export_runs(tmp_path / "runs.csv", tmp_path / "monthly.csv")
    runs = pd.read_csv(tmp_path / "runs.csv")
    monthly = pd.read_csv(tmp_path / "monthly.csv")
    assert sorted(runs["stage"]) == ["test", "validation"] and set(runs["run_name"]) == {"model_a"}
    assert (abs(runs["mae"] - 0.5) < 1e-12).all() and (runs["data_fingerprint"] == tracking.data_fingerprint(prices)).all()
    assert len(monthly) == 4  # 2 runs x 2 months (October and November 2025)
    assert set(monthly["month"]) == {"2025-10", "2025-11"}