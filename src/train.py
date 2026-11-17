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
from sklearn.model_selection import GridSearchCV

from pipeline import RANDOM_STATE, TARGET, candidate_models, save_model


class DateGroupedExpandingSplit:
    """Expanding chronological folds where each date belongs to one partition."""

    def __init__(self, n_splits: int = 4, validation_days: int = 28, minimum_train_days: int = 60):
        self.n_splits = n_splits
        self.validation_days = validation_days
        self.minimum_train_days = minimum_train_days

    def split(self, X, y=None, groups=None):
        dates = pd.to_datetime(X["date"]).dt.normalize()
        unique_dates = np.array(sorted(dates.unique()), dtype="datetime64[ns]")
        first_validation = len(unique_dates) - self.n_splits * self.validation_days
        if first_validation < self.minimum_train_days:
            raise ValueError("not enough unique dates for expanding validation folds")
        for fold in range(self.n_splits):
            start = first_validation + fold * self.validation_days
            stop = start + self.validation_days
            train_dates = set(unique_dates[:start])
            validation_dates = set(unique_dates[start:stop])
            train_indices = np.flatnonzero(dates.isin(train_dates).to_numpy())
            validation_indices = np.flatnonzero(dates.isin(validation_dates).to_numpy())
            yield train_indices, validation_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def safe_mape(actual: pd.Series, predicted: np.ndarray) -> float:
    actual_values = np.asarray(actual)
    mask = actual_values != 0
    return float(np.mean(np.abs((actual_values[mask] - predicted[mask]) / actual_values[mask])) * 100)


def metric_row(actual: pd.Series, predicted: np.ndarray, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}mae": float(mean_absolute_error(actual, predicted)),
        f"{prefix}rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        f"{prefix}r2": float(r2_score(actual, predicted)),
    }


def evaluate_folds(estimator, frame: pd.DataFrame, target: pd.Series, folds) -> tuple[dict, list[dict]]:
    fold_metrics = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        model = clone(estimator)
        model.fit(frame.iloc[train_indices], target.iloc[train_indices])
        prediction = np.maximum(model.predict(frame.iloc[validation_indices]), 1e-6)
        metric_values = metric_row(target.iloc[validation_indices], prediction)
        metrics = {
            "mae": metric_values["mae"],
            "rmse": metric_values["rmse"],
            "r2": metric_values["r2"],
        }
        metrics["fold"] = fold_number
        metrics["validation_start"] = str(frame.iloc[validation_indices]["date"].min().date())
        metrics["validation_end"] = str(frame.iloc[validation_indices]["date"].max().date())
        fold_metrics.append(metrics)
    summary = {
        "cv_mae": float(np.mean([item["mae"] for item in fold_metrics])),
        "cv_rmse": float(np.mean([item["rmse"] for item in fold_metrics])),
        "cv_r2": float(np.mean([item["r2"] for item in fold_metrics])),
        "latest_cv_mae": float(fold_metrics[-1]["mae"]),
    }
    return summary, fold_metrics


def write_residual_analysis(actual: pd.Series, prediction: np.ndarray, frame: pd.DataFrame, path: Path) -> None:
    residuals = actual.to_numpy() - prediction
    analysis = pd.DataFrame({
        "load_id": frame["load_id"].to_numpy(),
        "date": frame["date"].to_numpy(),
        "actual_rate": actual.to_numpy(),
        "predicted_rate": prediction,
        "absolute_error": np.abs(residuals),
        "rate_per_mile": actual.to_numpy() / frame["distance"].to_numpy(),
        "pickup": frame["pickup"].to_numpy(),
        "delivery": frame["delivery"].to_numpy(),
        "equipment": frame["equipment"].to_numpy(),
        "market_index": frame["market_index"].to_numpy(),
        "quote_signal": frame["quote_signal"].to_numpy(),
    })
    analysis.sort_values("absolute_error", ascending=False).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train-test.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.drop_duplicates(subset=["load_id"]).sort_values(["date", "load_id"]).reset_index(drop=True)
    unique_dates = np.sort(data["date"].dt.normalize().unique())
    split_index = int(np.ceil(len(unique_dates) * 0.8))
    cutoff_date = pd.Timestamp(unique_dates[split_index])
    train = data[data["date"] < cutoff_date].copy().reset_index(drop=True)
    holdout = data[data["date"] >= cutoff_date].copy().reset_index(drop=True)
    y_train = train[TARGET]
    y_holdout = holdout[TARGET]

    cv = DateGroupedExpandingSplit(n_splits=4, validation_days=28, minimum_train_days=60)
    folds = list(cv.split(train))
    results = []
    searches = {}
    holdout_predictions = {}
    fold_rows = []

    for name, (estimator, params) in candidate_models().items():
        started = time.perf_counter()
        search = GridSearchCV(
            estimator=estimator,
            param_grid=params,
            scoring="neg_mean_absolute_error",
            cv=folds,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(train.drop(columns=[TARGET]), y_train)
        best_estimator = search.best_estimator_
        cv_summary, fold_metrics = evaluate_folds(
            best_estimator, train.drop(columns=[TARGET]), y_train, folds
        )
        prediction = np.maximum(search.predict(holdout.drop(columns=[TARGET])), 1e-6)
        searches[name] = search
        holdout_predictions[name] = prediction
        row = {
            "model": name,
            **cv_summary,
            **metric_row(y_holdout, prediction, "holdout_"),
            "holdout_mape_percent": safe_mape(y_holdout, prediction),
            "training_time_seconds": time.perf_counter() - started,
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        }
        results.append(row)
        for metrics in fold_metrics:
            fold_rows.append({"model": name, **metrics})

    comparison = pd.DataFrame(results).sort_values(
        ["cv_mae", "latest_cv_mae", "cv_rmse"]
    ).reset_index(drop=True)
    report_path = Path(args.reports_dir)
    figure_path = report_path / "figures"
    report_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(report_path / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(report_path / "fold_metrics.csv", index=False)

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
            "cv": "DateGroupedExpandingSplit(n_splits=4, validation_days=28)",
            "selection": "mean chronological CV MAE, then latest CV MAE",
        }, file, indent=2)

    selected_prediction = holdout_predictions[best_name]
    write_residual_analysis(
        y_holdout,
        selected_prediction,
        holdout,
        report_path / "residual_analysis.csv",
    )
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
    print(f"Selected model by grouped CV: {best_name}")


if __name__ == "__main__":
    main()
