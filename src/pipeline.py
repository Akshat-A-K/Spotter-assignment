from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone


RANDOM_STATE = 42
TARGET = "posted_rate"
CATEGORICAL = ["pickup", "delivery", "equipment", "route"]
NUMERIC = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal", "weight_missing", "market_missing",
    "weight_abs", "distance_per_weight", "route_lat_delta", "route_lon_delta",
    "haversine_miles", "distance_ratio", "route_bearing_sin", "route_bearing_cos",
    "distance_log", "weight_log", "market_log", "market_quote_interaction",
    "market_days_interaction", "quote_distance_interaction",
    "day_of_year", "day_of_week", "month", "day", "days_since_start",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "distance_weight_interaction", "weight_per_distance",
    "route_frequency", "pickup_frequency", "delivery_frequency",
    "equipment_frequency", "route_equipment_frequency",
]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal"]:
        if column not in out:
            out[column] = np.nan
    date = pd.to_datetime(out["date"], errors="coerce")
    if date.isna().any():
        raise ValueError("date contains invalid values")
    out["pickup"] = out["pickup"].fillna("__MISSING__").astype(str)
    out["delivery"] = out["delivery"].fillna("__MISSING__").astype(str)
    out["equipment"] = out["equipment"].fillna("__MISSING__").astype(str)
    out["route"] = out["pickup"] + " -> " + out["delivery"]
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["market_index"] = pd.to_numeric(out["market_index"], errors="coerce")
    out["weight_missing"] = out["weight"].isna().astype(float)
    out["market_missing"] = out["market_index"].isna().astype(float)
    out["weight_abs"] = out["weight"].abs()
    out["distance_per_weight"] = out["distance"] / out["weight_abs"].clip(lower=1000)
    out["route_lat_delta"] = out["delivery_lat"] - out["pickup_lat"]
    out["route_lon_delta"] = out["delivery_lon"] - out["pickup_lon"]
    lat1 = np.radians(out["pickup_lat"])
    lat2 = np.radians(out["delivery_lat"])
    delta_lat = lat2 - lat1
    delta_lon = np.radians(out["delivery_lon"] - out["pickup_lon"])
    haversine_a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    out["haversine_miles"] = 3958.7613 * 2.0 * np.arcsin(np.sqrt(haversine_a.clip(0.0, 1.0)))
    out["distance_ratio"] = out["distance"] / out["haversine_miles"].clip(lower=1.0)
    out["route_bearing_sin"] = np.sin(np.arctan2(delta_lon, delta_lat))
    out["route_bearing_cos"] = np.cos(np.arctan2(delta_lon, delta_lat))
    out["distance_log"] = np.log1p(out["distance"].clip(lower=0.0))
    out["weight_log"] = np.log1p(out["weight_abs"].clip(lower=0.0))
    out["market_log"] = np.log1p(out["market_index"].clip(lower=0.0))
    out["day_of_year"] = date.dt.dayofyear
    out["day_of_week"] = date.dt.dayofweek
    out["month"] = date.dt.month
    out["day"] = date.dt.day
    out["days_since_start"] = (date - pd.Timestamp("2025-01-01")).dt.days
    out["day_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["distance_weight_interaction"] = out["distance"] * out["weight_abs"] / 100000
    out["weight_per_distance"] = out["weight_abs"] / out["distance"].clip(lower=1)
    out["market_quote_interaction"] = out["market_index"] * out["quote_signal"]
    out["market_days_interaction"] = out["market_index"] * out["days_since_start"]
    out["quote_distance_interaction"] = out["quote_signal"] * out["distance_log"]
    out["route_frequency"] = 0.0
    out["pickup_frequency"] = 0.0
    out["delivery_frequency"] = 0.0
    out["equipment_frequency"] = 0.0
    out["route_equipment_frequency"] = 0.0
    return out[CATEGORICAL + NUMERIC]


def numeric_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])


def categorical_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])


def preprocessing() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", numeric_preprocessor(), NUMERIC),
        ("categorical", categorical_preprocessor(), CATEGORICAL),
    ])


class FrequencyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, frame: pd.DataFrame, y=None):
        self.counts_ = {
            "route_frequency": frame["route"].value_counts(),
            "pickup_frequency": frame["pickup"].value_counts(),
            "delivery_frequency": frame["delivery"].value_counts(),
            "equipment_frequency": frame["equipment"].value_counts(),
            "route_equipment_frequency": (
                frame["route"] + " | " + frame["equipment"]
            ).value_counts(),
        }
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["route_frequency"] = out["route"].map(self.counts_["route_frequency"]).fillna(0).astype(float)
        out["pickup_frequency"] = out["pickup"].map(self.counts_["pickup_frequency"]).fillna(0).astype(float)
        out["delivery_frequency"] = out["delivery"].map(self.counts_["delivery_frequency"]).fillna(0).astype(float)
        out["equipment_frequency"] = out["equipment"].map(self.counts_["equipment_frequency"]).fillna(0).astype(float)
        route_equipment = out["route"] + " | " + out["equipment"]
        out["route_equipment_frequency"] = route_equipment.map(
            self.counts_["route_equipment_frequency"]
        ).fillna(0).astype(float)
        return out


class MarketAdjustedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator=None, clean_low=1.2, clean_high=5.0, level_alpha=10.0):
        self.base_estimator = base_estimator
        self.clean_low = clean_low
        self.clean_high = clean_high
        self.level_alpha = level_alpha

    @staticmethod
    def _level_features(frame: pd.DataFrame, market_mean: float, columns=None) -> pd.DataFrame:
        result = pd.DataFrame(index=frame.index)
        market = frame["market_index"] if "market_index" in frame else pd.Series(market_mean, index=frame.index)
        result["market"] = pd.to_numeric(market, errors="coerce").fillna(market_mean)
        result["days"] = (pd.to_datetime(frame["date"]) - pd.Timestamp("2025-01-01")).dt.days
        result["month"] = pd.to_datetime(frame["date"]).dt.month
        result = pd.get_dummies(result, columns=["month"], dtype=float)
        if columns is not None:
            result = result.reindex(columns=columns, fill_value=0.0)
        return result.astype(float)

    def _daily_table(self, frame: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        daily = frame[["date", "market_index", "distance"]].copy()
        daily["log_rate_per_mile"] = np.log(target.to_numpy() / daily["distance"].clip(lower=1).to_numpy())
        daily["clean"] = np.exp(daily["log_rate_per_mile"]).between(self.clean_low, self.clean_high)
        daily = daily[daily["clean"]].copy()
        return daily.groupby("date", as_index=False).agg(
            level=("log_rate_per_mile", "median"),
            market=("market_index", "mean"),
        )

    def fit(self, frame: pd.DataFrame, target: pd.Series):
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        self.market_mean_ = float(pd.to_numeric(data["market_index"], errors="coerce").mean())
        daily = self._daily_table(data, target)
        daily_frame = daily.rename(columns={"market": "market_index"})
        self.level_columns_ = list(self._level_features(daily_frame, self.market_mean_).columns)
        rolling_levels = {}
        for date in daily["date"].sort_values():
            prior = daily[daily["date"] < date]
            if len(prior) < 20:
                rolling_levels[date] = float(prior["level"].median() if len(prior) else daily["level"].median())
                continue
            prior_frame = prior.rename(columns={"market": "market_index"})
            features = self._level_features(prior_frame, self.market_mean_, self.level_columns_)
            level_model = Ridge(alpha=self.level_alpha).fit(features, prior["level"])
            current = pd.DataFrame([{"date": date, "market_index": daily.loc[daily["date"] == date, "market"].iloc[0]}])
            current_features = self._level_features(current, self.market_mean_, self.level_columns_)
            rolling_levels[date] = float(level_model.predict(current_features)[0])
        train_levels = data["date"].map(rolling_levels).fillna(float(daily["level"].median())).to_numpy()
        rates = target.to_numpy() / data["distance"].clip(lower=1).to_numpy()
        clean = (rates >= self.clean_low) & (rates <= self.clean_high)
        self.level_model_ = Ridge(alpha=self.level_alpha).fit(
            self._level_features(daily_frame, self.market_mean_, self.level_columns_),
            daily["level"],
        )
        residual_target = np.log(rates) - train_levels
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(data.loc[clean], pd.Series(residual_target[clean], index=data.index[clean]))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"])
        levels = self.level_model_.predict(
            self._level_features(data, self.market_mean_, self.level_columns_)
        )
        residual = self.base_estimator_.predict(data)
        return np.maximum(np.exp(residual + levels) * data["distance"].to_numpy(), 1e-6)


def standard_model(estimator) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(make_features, validate=False)),
        ("frequency_features", FrequencyFeatures()),
        ("preprocess", preprocessing()),
        ("model", estimator),
    ])


def catboost_model() -> Pipeline:
    estimator = CatBoostRegressor(
        loss_function="MAE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
        cat_features=tuple(CATEGORICAL),
    )
    return Pipeline([
        ("features", FunctionTransformer(make_features, validate=False)),
        ("frequency_features", FrequencyFeatures()),
        ("model", estimator),
    ])


def candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "Ridge": (
            Pipeline([
                ("features", FunctionTransformer(make_features, validate=False)),
                ("frequency_features", FrequencyFeatures()),
                ("preprocess", preprocessing()),
                ("scale", StandardScaler()),
                ("model", Ridge()),
            ]),
            {"model__alpha": [1.0, 10.0, 30.0]},
        ),
        "DecisionTreeRegressor": (
            standard_model(DecisionTreeRegressor(random_state=RANDOM_STATE)),
            {"model__max_depth": [8, 16], "model__min_samples_leaf": [2, 10], "model__ccp_alpha": [0.0, 0.001]},
        ),
        "LinearSVR": (
            Pipeline([
                ("features", FunctionTransformer(make_features, validate=False)),
                ("preprocess", preprocessing()),
                ("scale", StandardScaler()),
                ("model", LinearSVR(random_state=RANDOM_STATE, max_iter=10000, dual="auto")),
            ]),
            {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.1, 10.0]},
        ),
        "ExtraTreesRegressor": (
            standard_model(ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1)),
            {"model__n_estimators": [300], "model__min_samples_leaf": [1, 2], "model__max_features": [0.8]},
        ),
        "RandomForestRegressor": (
            standard_model(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
            {"model__n_estimators": [300], "model__min_samples_leaf": [1, 2], "model__max_features": [0.8]},
        ),
        "HistGradientBoostingRegressor": (
            standard_model(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            {"model__max_iter": [250], "model__learning_rate": [0.04, 0.06], "model__max_leaf_nodes": [15, 31], "model__l2_regularization": [0.5]},
        ),
        "CatBoostRegressor": (
            catboost_model(),
            {"model__iterations": [500], "model__depth": [6, 8], "model__learning_rate": [0.03], "model__l2_leaf_reg": [3]},
        ),
        "LGBMRegressor": (
            standard_model(LGBMRegressor(objective="regression_l1", random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)),
            {
                "model__n_estimators": [300, 600],
                "model__learning_rate": [0.03],
                "model__num_leaves": [31, 63],
                "model__max_depth": [-1],
                "model__min_child_samples": [30],
                "model__reg_lambda": [5, 10],
                "model__reg_alpha": [0, 1],
            },
        ),
        "MarketAdjustedLGBM": (
            MarketAdjustedRegressor(
                base_estimator=standard_model(LGBMRegressor(
                    objective="regression_l1",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=-1,
                ))
            ),
            {
                "base_estimator__model__n_estimators": [600],
                "base_estimator__model__learning_rate": [0.03],
                "base_estimator__model__num_leaves": [31],
                "base_estimator__model__min_child_samples": [30],
                "base_estimator__model__reg_lambda": [5],
                "base_estimator__model__reg_alpha": [0],
                "clean_low": [1.2],
                "clean_high": [5.0],
            },
        ),
        "XGBRegressor": (
            standard_model(XGBRegressor(objective="reg:absoluteerror", random_state=RANDOM_STATE, n_jobs=1, tree_method="hist")),
            {"model__n_estimators": [300], "model__learning_rate": [0.03, 0.05], "model__max_depth": [4, 6], "model__min_child_weight": [1], "model__subsample": [0.8], "model__colsample_bytree": [0.8], "model__reg_lambda": [1]},
        ),
    }


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    if "pipeline" not in sys.modules:
        sys.modules["pipeline"] = sys.modules[__name__]
    return joblib.load(path)
