# 2–3 minute Loom script

**Opening (15 seconds).** I built a reproducible freight-rate regression
pipeline for 12,000 future Spotter loads. The final model is refit on all
48,000 labeled development rows.

**Data quality (30 seconds).** The data contains missing weights and market
indices, negative weights, unseen cities, and 736 unseen validation routes. I
retained every row, added missingness and absolute-value features, and fitted
imputation and frequency encodings inside each training fold. The validation
set is never used to fit target-derived information.

**Validation (30 seconds).** The previous row-based time split could place the
same date in both sides of a fold. I replaced it with four expanding
chronological monthly windows of 28 complete dates. The latest 20% of dates,
September 2 through October 31, remains an untouched final holdout.

**Models (35 seconds).** I compared regularized linear, SVM, single-tree,
random-forest, extra-trees, histogram-gradient, CatBoost, XGBoost, standard
LightGBM, and a market-adjusted LightGBM. The market-adjusted model estimates a
prior-date market level for log rate per mile and learns the load-level residual.
It won grouped CV with MAE 99.76 and achieved holdout MAE 103.15, versus 108.15
for XGBoost and 109.15 for standard LightGBM.

**Extreme errors (25 seconds).** The high-rate observations are distributed
across Dry Van, Reefer, and Flatbed, so I did not cap or delete them. The largest
1% of holdout residuals account for 97.4% of squared error; those rows are
preserved in `reports/residual_analysis.csv` for auditability.

**Outputs (25 seconds).** `src/predict.py` writes
`validation_predictions.csv` and a separate `december_predictions.csv`. It
preserves template ID order, checks finite positive predictions, and never
overwrites `december-chart-inputs.csv`. Finally, the supplied `score.py`
validates both files and generates the December chart.
