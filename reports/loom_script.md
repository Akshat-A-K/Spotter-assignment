# 2–3 minute Loom script

**Opening (15 seconds).** I built a reproducible freight-rate regression pipeline.
The target is `posted_rate`; the final deliverable contains one positive prediction
for each of the 12,000 validation loads.

**Data and quality (35 seconds).** The development data contains 48,000 rows and
14 columns. It has pickup and delivery locations, coordinates, distance, equipment,
weight, date, market index, quote signal, and the target. There are no duplicate
rows. Weight and market index have missing values, and the data contains negative
weights, so I retained every row, added missingness indicators, median-imputed
numeric values using training data only, and added an absolute-weight feature rather
than silently dropping observations.

**Features and validation (35 seconds).** I created a route feature, route coordinate
deltas, distance-per-weight, date components, cyclic day/month seasonality, and
equipment/location indicators. Because the final prediction period is later than the
development data, I sorted by date and used the latest 20% as a completely held-out
chronological validation fold. The cutoff was 2025-08-31. This better tests future
generalization than a random split.

**Models and selection (35 seconds).** I compared Ridge and two
HistGradientBoosting settings using a small time-series grid search. The selected
HistGradientBoosting model performed best on the holdout with MAE 123.42, RMSE
632.15, and R² 0.8278, so I refit it on all 48,000 labeled rows. The comparison
table and residual plots are in the reports directory.

**Outputs (25 seconds).** The saved model predicts the validation template in exact
load ID order and writes `validation_predictions.csv`. It also predicts the 31
fixed December inputs. Finally, I run the supplied `score.py`; it validates both
schemas and generates `reports/figures/candidate_december.png`. The scorer passed
with 12,000 validation predictions and 31 December predictions.

**Close (10 seconds).** The workflow is reproducible with the commands in README.md,
and the PDF report documents the data, methodology, actual results, limitations, and
the generated chart.
