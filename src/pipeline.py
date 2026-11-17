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
from sklearn.base import BaseEstimator, TransformerMixin


RANDOM_STATE = 42
TARGET = "posted_rate"
CATEGORICAL = ["pickup", "delivery", "equipment", "route"]
NUMERIC = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal", "weight_missing", "market_missing",
    "weight_abs", "distance_per_weight", "route_lat_delta", "route_lon_delta",
    "day_of_year", "day_of_week", "month", "day", "days_since_start",
    "day_sin", "day_cos", "month_sin", "month_cos",
    "distance_weight_interaction", "weight_per_distance",
    "route_frequency",
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
    out["route_frequency"] = 0.0
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


class RouteFrequency(BaseEstimator, TransformerMixin):
    def fit(self, frame: pd.DataFrame, y=None):
        self.counts_ = frame["route"].value_counts()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["route_frequency"] = out["route"].map(self.counts_).fillna(0).astype(float)
        return out


def standard_model(estimator) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(make_features, validate=False)),
        ("route_frequency", RouteFrequency()),
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
        ("route_frequency", RouteFrequency()),
        ("model", estimator),
    ])


def candidate_models() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "Ridge": (
            Pipeline([
                ("features", FunctionTransformer(make_features, validate=False)),
                ("route_frequency", RouteFrequency()),
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
