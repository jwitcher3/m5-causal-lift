from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import polars as pl


def _ridge_weights(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    k = X.shape[1]
    A = (X.T @ X) + alpha * np.eye(k)
    b = X.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _align_columns(prev: pl.DataFrame, cur: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Align schemas for concat, casting NULL columns to target dtypes (older Polars-safe)."""
    prev_schema = prev.schema
    cur_schema = cur.schema
    all_cols = sorted(set(prev_schema.keys()) | set(cur_schema.keys()))

    # prefer current dtype if present
    target = {c: (cur_schema.get(c) or prev_schema.get(c) or pl.Null) for c in all_cols}

    def coerce(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
        for c in all_cols:
            dt = target[c]
            if c not in schema:
                df = df.with_columns(pl.lit(None).cast(dt).alias(c))
            else:
                if schema[c] == pl.Null and dt != pl.Null:
                    df = df.with_columns(pl.col(c).cast(dt))
                elif schema[c] != dt and dt != pl.Null:
                    df = df.with_columns(pl.col(c).cast(dt))
        return df.select(all_cols)

    return coerce(prev, prev_schema), coerce(cur, cur_schema)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--campaign_id", type=str, default="cmp_001")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--use_log1p", action="store_true")
    p.add_argument("--donor_grain", type=str, default="store", choices=["store", "store_dept"])
    args = p.parse_args()

    processed = Path(args.processed_dir)
    gt_path = processed / "fact_ground_truth.parquet"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}. Run simulator first.")

    gt = pl.read_parquet(gt_path).filter(pl.col("campaign_id") == args.campaign_id)

    # Treated aggregate time series
    treated_ts = (
        gt.filter(pl.col("treated") == 1)
          .group_by("date")
          .agg(
              pl.col("y_obs").sum().alias("y_treated"),
              pl.col("tau").sum().alias("tau_true"),
              pl.col("in_campaign").max().alias("in_campaign"),
          )
          .sort("date")
    )

    # True ATT at aggregate level: average daily lift in treated aggregate series during campaign
    att_true = float(treated_ts.filter(pl.col("in_campaign") == 1).select(pl.col("tau_true").mean()).item())

    # Donor pool: controls
    controls = gt.filter(pl.col("treated") == 0)

    if args.donor_grain == "store":
        donors = controls.group_by(["store_id", "date"]).agg(pl.col("y_obs").sum().alias("y"))
        donor_id_cols = ["store_id"]
    else:
        donors = controls.group_by(["store_id", "dept_id", "date"]).agg(pl.col("y_obs").sum().alias("y"))
        donor_id_cols = ["store_id", "dept_id"]

    donors = donors.with_columns(
        pl.concat_str([pl.col(c).cast(pl.Utf8) for c in donor_id_cols], separator="|").alias("donor_id")
    )

    # Pivot donors wide
    donor_wide = (
        donors.pivot(values="y", index="date", on="donor_id", aggregate_function="first")
              .sort("date")
              .fill_null(0)
    )

    panel = (
        treated_ts.select(["date", "y_treated", "in_campaign"])
        .join(donor_wide, on="date", how="left")
        .fill_null(0)
        .sort("date")
    )

    donor_cols = [c for c in panel.columns if c not in ("date", "y_treated", "in_campaign")]

    pre = panel.filter(pl.col("in_campaign") == 0)

    y_all_units = panel["y_treated"].to_numpy().astype(np.float64)
    X_all_units = panel.select(donor_cols).to_numpy().astype(np.float64)

    y_pre_units = pre["y_treated"].to_numpy().astype(np.float64)
    X_pre_units = pre.select(donor_cols).to_numpy().astype(np.float64)

    if args.use_log1p:
        y_pre = np.log1p(y_pre_units)
        X_pre = np.log1p(X_pre_units)
        X_all_fit = np.log1p(X_all_units)
    else:
        y_pre = y_pre_units
        X_pre = X_pre_units
        X_all_fit = X_all_units
w = _ridge_weights(X_pre, y_pre, alpha=args.alpha)

y0_hat_fit = X_all_fit @ w
y0_hat = np.expm1(y0_hat_fit) if args.use_log1p else y0_hat_fit

in_c = panel["in_campaign"].to_numpy().astype(np.int64)
att_hat = float(np.mean((y_all_units - y0_hat)[in_c == 1]))

# RMSE on the fit scale (raw or log)
rmse_pre = float(np.sqrt(np.mean((y_pre - (X_pre @ w)) ** 2)))


res = pl.DataFrame([{
        "campaign_id": args.campaign_id,
        "method": f"scm_ridge_{args.donor_grain}",
        "att_hat": att_hat,
        "att_true": att_true,
        "bias": att_hat - att_true,
        "alpha": args.alpha,
        "rmse_pre": rmse_pre,
        "n_dates": int(panel.height),
        "n_donors": int(len(donor_cols)),
    }])

out_results = processed / "fact_method_results.parquet"
if out_results.exists():
        prev = pl.read_parquet(out_results)
        prev2, res2 = _align_columns(prev, res)
        out = pl.concat([prev2, res2], how="vertical").unique(subset=["campaign_id", "method"], keep="last")
    else:
        out = res

    out.write_parquet(out_results)

    # Save series for plotting/dashboard later
ts = (
        panel.select(["date", "y_treated", "in_campaign"])
        .with_columns(pl.Series("y0_hat", y0_hat.tolist()).cast(pl.Float64))
        .with_columns((pl.col("y_treated") - pl.col("y0_hat")).alias("lift_hat"))
    )
ts.write_parquet(processed / f"scm_series_{args.campaign_id}_{args.donor_grain}.parquet")

print(res)


if __name__ == "__main__":
    main()
