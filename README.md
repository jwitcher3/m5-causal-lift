# m5-causal-lift

Portfolio project: simulate synthetic ecommerce campaigns on M5 data (ground-truth lift),
estimate lift with DiD/event study + synthetic control, and evaluate vs truth.

## Setup
pip install "polars[all]" pyarrow

## Build processed tables
Put `calendar.csv` and `sales_train_validation.csv` in `data/raw/`, then run:

python src/m5lift/io/build_processed.py --raw_dir data/raw --out_dir data/processed --grain store_dept
