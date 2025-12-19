# M5 Causal Lift (Incrementality Portfolio Project)

Goal: simulate synthetic marketing campaigns with known ground-truth lift on top of M5 data,
then estimate lift using DiD/event study and synthetic control, and evaluate against truth.

## Quickstart
1) Put M5 CSVs in `data/raw/` (calendar.csv, sales_train_validation.csv, sell_prices.csv).
2) Build processed tables:
   python src/m5lift/io/build_processed.py --raw_dir data/raw --out_dir data/processed --grain store_dept
