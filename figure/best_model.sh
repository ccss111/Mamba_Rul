#!/usr/bin/env bash
set -e

PYTHONPATH=/CMAPSS-release python scipt/test_model.py --sub-dataset FD001 --model-path /CMAPSS-release/logs/FD001_FD001_F/seed51/model_FD001_0420_235029_seed51.pkl --no-cuda --save-pred-svg --pred-svg-path /CMAPSS-release/figure/FD001_pred.svg --save-pred-csv --pred-csv-path /CMAPSS-release/figure/FD001_pred.csv --smooth-rate 30

PYTHONPATH=/CMAPSS-release python scipt/test_model.py --sub-dataset FD002 --model-path /CMAPSS-release/logs/FD002_FD002_F/seed33/model_FD002_0421_003026_seed33.pkl --no-cuda --save-pred-svg --pred-svg-path /CMAPSS-release/figure/FD002_pred.svg --save-pred-csv --pred-csv-path /CMAPSS-release/figure/FD002_pred.csv --smooth-rate 40

PYTHONPATH=/CMAPSS-release python scipt/test_model.py --sub-dataset FD003 --model-path /CMAPSS-release/logs/FD003_FD003_F/seed30/model_FD003_0421_012044_seed30.pkl --no-cuda --save-pred-svg --pred-svg-path /CMAPSS-release/figure/FD003_pred.svg --save-pred-csv --pred-csv-path /CMAPSS-release/figure/FD003_pred.csv --smooth-rate 30

PYTHONPATH=/CMAPSS-release python scipt/test_model.py --sub-dataset FD004 --model-path /CMAPSS-release/logs/FD004_FD004_F/seed33/model_FD004_0421_021453_seed33.pkl --no-cuda --save-pred-svg --pred-svg-path /CMAPSS-release/figure/FD004_pred.svg --save-pred-csv --pred-csv-path /CMAPSS-release/figure/FD004_pred.csv --smooth-rate 40

python /CMAPSS-release/figure/combine_four_predictions.py \
	--fd001-csv /CMAPSS-release/figure/FD001_pred.csv \
	--fd002-csv /CMAPSS-release/figure/FD002_pred.csv \
	--fd003-csv /CMAPSS-release/figure/FD003_pred.csv \
	--fd004-csv /CMAPSS-release/figure/FD004_pred.csv \
	--output /CMAPSS-release/figure/FD001_FD004_combined.svg
