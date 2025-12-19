from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--grain", type=str, default="store_dept", choices=["store_dept", "store_item"])
    args = p.parse_args()

    processed = Path(args.processed_dir)
    fact_path = processed / "fact_sales_daily.parquet"
    if not fact_path.exists():
        raise FileNotFoundError(f"Missing {fact_path}. Run build_processed.py first.")

    df = pl.read_parquet(fact_path)

    # keys depend on grain
    keys = ["store_id", "dept_id"] if args.grain == "store_dept" else ["store_id", "item_id"]

    # ensure date typed + sorted (required for correct windows)
    if df["date"].dtype != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    df = df.sort(keys + ["date"])

    y = "units"

    feat = df.with_columns(
        [
            pl.col(y).shift(1).over(keys).alias("lag_1"),
            pl.col(y).shift(7).over(keys).alias("lag_7"),
            pl.col(y).shift(14).over(keys).alias("lag_14"),
            pl.col(y).shift(28).over(keys).alias("lag_28"),
            # shift(1) first = no leakage into rolling windows
            pl.col(y).shift(1).over(keys).rolling_mean(7).alias("roll_mean_7"),
            pl.col(y).shift(1).over(keys).rolling_mean(28).alias("roll_mean_28"),
            pl.col(y).shift(1).over(keys).rolling_sum(7).alias("roll_sum_7"),
            pl.col(y).shift(1).over(keys).rolling_sum(28).alias("roll_sum_28"),
        ]
    ).select(
        keys
        + ["date", "units", "lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_sum_7", "roll_sum_28"]
    )

    out_path = processed / "feat_sales_lags.parquet"
    feat.write_parquet(out_path)

    print(f"Wrote {out_path} rows={feat.height:,} cols={len(feat.columns)}")


if __name__ == "__main__":
    main()
