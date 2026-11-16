from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from pipeline import RANDOM_STATE, TARGET, candidate_models, save_model


def safe_mape(actual: pd.Series, predicted: np.ndarray) -> float:
    actual_values = np.asarray(actual)
    mask = actual_values != 0
    return float(np.mean(np.abs((actual_values[mask] - predicted[mask]) / actual_values[mask])) * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train-test.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.drop_duplicates(subset=["load_id"]).sort_values(["date", "load_id"]).reset_index(drop=True)
    unique_dates = np.sort(data["date"].unique())
    split_index = int(np.ceil(len(unique_dates) * 0.8))
    cutoff_date = pd.Timestamp(unique_dates[split_index])
    train = data[data["date"] < cutoff_date].copy()
    holdout = data[data["date"] >= cutoff_date].copy()
    y_train = train[TARGET]
    y_holdout = holdout[TARGET]
    cv = TimeSeriesSplit(n_splits=5)
    results = []
    holdout_predictions = {}
    searches = {}

    for name, (estimator, params) in candidate_models().items():
        started = time.perf_counter()
        search = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(train.drop(columns=[TARGET]), y_train)
        prediction = np.maximum(search.predict(holdout.drop(columns=[TARGET])), 1e-6)
        searches[name] = search
        holdout_predictions[name] = prediction
        results.append({
            "model": name,
            "cv_mae": -float(search.best_score_),
            "holdout_mae": mean_absolute_error(y_holdout, prediction),
            "holdout_rmse": mean_squared_error(y_holdout, prediction) ** 0.5,
            "holdout_r2": r2_score(y_holdout, prediction),
            "holdout_mape_percent": safe_mape(y_holdout, prediction),
            "training_time_seconds": time.perf_counter() - started,
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        })

    comparison = pd.DataFrame(results).sort_values(["holdout_mae", "holdout_rmse"]).reset_index(drop=True)
    report_path = Path(args.reports_dir)
    figure_path = report_path / "figures"
    report_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_path / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    best_params = searches[best_name].best_params_
    final_estimator = clone(candidate_models()[best_name][0]).set_params(**best_params)
    final_estimator.fit(data.drop(columns=[TARGET]), data[TARGET])
    save_model(final_estimator, Path(args.model_dir) / "best_model.joblib")
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.model_dir) / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump({
            "model": best_name,
            "best_params": best_params,
            "random_state": RANDOM_STATE,
            "development_start": str(data["date"].min().date()),
            "development_end": str(data["date"].max().date()),
            "training_start": str(train["date"].min().date()),
            "training_end": str(train["date"].max().date()),
            "holdout_start": str(holdout["date"].min().date()),
            "holdout_end": str(holdout["date"].max().date()),
            "cutoff_date": str(cutoff_date.date()),
            "cv": "TimeSeriesSplit(n_splits=5)",
        }, file, indent=2)

    selected_prediction = holdout_predictions[best_name]
    residuals = y_holdout.to_numpy() - selected_prediction
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(y_holdout, selected_prediction, s=5, alpha=0.25, color="#064A56")
    limits = [min(y_holdout.min(), selected_prediction.min()), max(y_holdout.max(), selected_prediction.max())]
    axes[0].plot(limits, limits, "--", color="#C4513B")
    axes[0].set(xlabel="Actual rate", ylabel="Predicted rate", title="Actual vs predicted")
    axes[1].hist(residuals, bins=60, color="#0B7285", alpha=0.85)
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
    axes[2].scatter(selected_prediction, residuals, s=5, alpha=0.25, color="#064A56")
    axes[2].axhline(0, color="#C4513B", linestyle="--")
    axes[2].set(xlabel="Predicted rate", ylabel="Actual - predicted", title="Residual vs predicted")
    fig.tight_layout()
    fig.savefig(figure_path / "residual_diagnostics.png", dpi=180)
    plt.close(fig)
    print(comparison.to_string(index=False))
    print(f"Training dates: {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Holdout dates: {holdout['date'].min().date()} to {holdout['date'].max().date()}")
    print(f"Selected model: {best_name}")


if __name__ == "__main__":
    main()
