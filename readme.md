# Spotter Freight Rate Prediction

This repository predicts `posted_rate` for Spotter freight loads. The final model
is a market-adjusted LightGBM regressor selected with grouped chronological
validation and refit on all labeled development rows.

## Data

The supplied files must be present in the repository root:

- `train-test.csv`: 48,000 labeled rows from January 1 through October 31, 2025.
- `validation.csv`: 12,000 unlabeled rows.
- `validation-predictions-template.csv`: required output IDs and schema.
- `december-chart-inputs.csv`: 31 fixed December chart inputs; never overwrite it.
- `score.py`: official output validator and chart generator.

## Reproduction

Use Python 3.10+:

```bash
python -m pip install -r requirements.txt
python src/train.py --data train-test.csv
python src/predict.py
python score.py --predictions validation_predictions.csv ^
  --december-predictions december_predictions.csv --output-dir reports/figures
```

On Windows, `run.bat` runs the same workflow. The final prediction files are
`validation_predictions.csv` and `december_predictions.csv`.

## Validation and model selection

The latest 20% of unique development dates is kept as an untouched holdout:
January 1 through September 1 are training/tuning dates, and September 2 through
October 31 are final holdout dates. Before that holdout, selection uses four
expanding monthly validation windows of 28 complete dates each. A date never
appears in both training and validation in any fold. `reports/fold_metrics.csv`
contains MAE, RMSE, and R2 for every model and fold.

The experiment compares regularized Ridge, LinearSVR, DecisionTree,
RandomForest, ExtraTrees, HistGradientBoosting, CatBoost, standard LightGBM,
XGBoost, and market-adjusted LightGBM. The selected model is chosen by mean
chronological CV MAE, then latest-window MAE. The actual run selected
MarketAdjustedLGBM:

- grouped CV MAE: **99.76**
- latest grouped-CV MAE: **103.59**
- untouched holdout MAE: **103.15**
- untouched holdout RMSE: **633.96**
- untouched holdout R2: **0.8280**

Under the same corrected validation, XGBoost reached holdout MAE 108.15 and
standard LightGBM 109.15. Results for all models and hyperparameters are in
`reports/model_comparison.csv`.

## Features and data quality

The pipeline retains all observations, including 300 missing weights, 374 missing
market-index values, and 292 negative weights in training. It uses missingness
indicators, absolute weight, fold-fitted median imputation, route/city/equipment
frequencies, haversine distance, geographic-distance ratio, route direction,
date seasonality, and market/quote/weight/distance interactions. Frequency
features are fitted inside each training fold and never use target values.
Unseen validation cities and 736 unseen routes are handled safely by the
categorical encoders and zero frequency.

The extreme-rate rows were not capped or removed. The 67 holdout rows above
$5/mile include 44 Dry Van, 16 Reefer, and 7 Flatbed rows, with no single
equipment explanation. The largest 1% of holdout absolute residuals account for
97.4% of squared error; `reports/residual_analysis.csv` preserves the detailed
evidence for investigation.

## Outputs

- `validation_predictions.csv`: exactly `load_id,predicted_rate`, 12,000 rows.
- `december_predictions.csv`: 31 fixed-input rows with predicted rates.
- `reports/model_comparison.csv`: aggregate model and holdout metrics.
- `reports/fold_metrics.csv`: fold-level chronological metrics.
- `reports/residual_analysis.csv`: holdout residual diagnostics.
- `reports/figures/residual_diagnostics.png`: residual plots.
- `reports/figures/candidate_december.png`: chart from the supplied `score.py`.
- `models/best_model.joblib`: serialized final model.
- `models/model_metadata.json`: selection and date metadata.

The prediction script validates positive finite predictions, preserves template
ID order, and does not modify `december-chart-inputs.csv`.
