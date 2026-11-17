from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--validation", default="validation.csv")
    parser.add_argument("--template", default="validation-predictions-template.csv")
    parser.add_argument("--output", default="validation_predictions.csv")
    parser.add_argument("--december", default="december-chart-inputs.csv")
    parser.add_argument("--december-output", default="december_predictions.csv")
    args = parser.parse_args()
    model = load_model(Path(args.model))
    validation = pd.read_csv(args.validation)
    template = pd.read_csv(args.template)
    if len(validation) != 12000 or len(template) != 12000:
        raise ValueError("validation data and template must each contain 12,000 rows")
    if "load_id" in validation and validation["load_id"].tolist() != template["load_id"].tolist():
        raise ValueError("validation rows and template IDs are not in the same order")
    if template["load_id"].duplicated().any():
        raise ValueError("template contains duplicate load IDs")
    validation_features = validation.drop(columns=["load_id", "predicted_rate"], errors="ignore")
    predictions = np.asarray(model.predict(validation_features), dtype=float)
    if predictions.shape != (12000,) or not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise ValueError("validation predictions must be finite and strictly positive")
    template["predicted_rate"] = predictions
    template[["load_id", "predicted_rate"]].to_csv(args.output, index=False)
    december = pd.read_csv(args.december)
    expected_columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(december.columns) != expected_columns or len(december) != 31:
        raise ValueError("December input does not match the required seven-column schema")
    december_predictions = np.asarray(model.predict(december.drop(columns=["predicted_rate"])), dtype=float)
    if not np.isfinite(december_predictions).all() or (december_predictions <= 0).any():
        raise ValueError("December predictions must be finite and strictly positive")
    december_output = december.copy()
    december_output["predicted_rate"] = december_predictions
    december_output.to_csv(args.december_output, index=False)
    print(f"Wrote {len(template):,} predictions to {args.output}")
    print(f"Wrote {len(december_output)} December predictions to {args.december_output}")


if __name__ == "__main__":
    main()
