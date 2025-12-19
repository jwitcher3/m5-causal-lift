from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def build_dim_calendar(calendar_csv: Path) -> pl.DataFrame:
    cal = pl.read_csv(calendar_csv)

    cal = cal.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date")
    ).with_columns(
        [
            (pl.col("date").dt.weekday() >= 5).alias("is_weekend"),
            (
                pl.col("event_name_1").is_not_null().cast(pl.Int8)
                + pl.col("event_name_2").is_not_null().cast(pl.Int8)
            ).alias("event_count"),
        ]
    )

    keep = [
        "date",
        "d",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "is_weekend",
        "event_count",
    ]
    return cal.select([c for c in keep if c in cal.columns]).sort("date")


def build_fact_sales_daily(
    sales_csv: Path,
    dim_calendar: pl.DataFrame,
    out_grain: str = "store_dept",
) -> pl.DataFrame:
    header_cols = pl.read_csv(sales_csv, n_rows=0).columns
    d_cols = [c for c in header_cols if c.startswith("d_")]

    id_vars = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    if "id" in header_cols:
        id_vars = ["id"] + id_vars

    lf = pl.scan_csv(sales_csv).select(id_vars + d_cols)

    lf_long = lf.melt(
        id_vars=id_vars,
        value_vars=d_cols,
        variable_name="d",
        value_name="units",
    )

    cal_lf = dim_calendar.lazy().select(["d", "date"])
    lf_joined = lf_long.join(cal_lf, on="d", how="left")

    if out_grain == "store_dept":
        group_keys = ["store_id", "dept_id", "date"]
    elif out_grain == "store_item":
        group_keys = ["store_id", "item_id", "date"]
    else:
        raise ValueError("out_grain must be one of: store_dept, store_item")

    fact = (
        lf_joined.group_by(group_keys)
        .agg(pl.col("units").sum().alias("units"))
        .with_columns(pl.col("units").cast(pl.Int64))
        .sort(group_keys)
        .collect()
    )

    return fact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, default="data/raw")
    p.add_argument("--out_dir", type=str, default="data/processed")
    p.add_argument("--grain", type=str, default="store_dept", choices=["store_dept", "store_item"])
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calendar_csv = raw_dir / "calendar.csv"
    sales_csv = raw_dir / "sales_train_validation.csv"

    if not calendar_csv.exists():
        raise FileNotFoundError(f"Missing {calendar_csv}")
    if not sales_csv.exists():
        raise FileNotFoundError(f"Missing {sales_csv}")

    dim_calendar = build_dim_calendar(calendar_csv)
    dim_calendar.write_parquet(out_dir / "dim_calendar.parquet")

    fact_sales = build_fact_sales_daily(sales_csv, dim_calendar, out_grain=args.grain)
    fact_sales.write_parquet(out_dir / "fact_sales_daily.parquet")

    print("Wrote:")
    print(f" - {out_dir / 'dim_calendar.parquet'}  rows={dim_calendar.height:,}")
    print(f" - {out_dir / 'fact_sales_daily.parquet'} rows={fact_sales.height:,}")


if __name__ == "__main__":
    main()
