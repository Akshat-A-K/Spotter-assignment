@echo off
python -m pip install -r requirements.txt
python src/train.py --data train-test.csv
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv --output-dir reports/figures
