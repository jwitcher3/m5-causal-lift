from __future__ import annotations

import argparse
from pathlib import Path
from datetime import timedelta, date as dt_date

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--campaign_id", type=str, default="cmp_001")
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--use_log1p", action="store_true")
    p.add_argument("--donor_grain", type=str, default="store", choices=["store", "store_dept"])
    p.add_argument("--n_placebos", type=int, default=50)
    p.add_argument("--min_pre_days", type=int, default=28)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    processed = Path(args.processed_dir)
    gt_path = processed / "fact_ground_truth.parquet"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}. Run simulator first.")

    gt = (
        pl.read_parquet(gt_path)
        .filter(pl.col("campaign_id") == args.campaign_id)
        .with_columns(pl.col("date").cast(pl.Date))
    )
    if gt.is_empty():
        raise ValueError(f"No rows found for campaign_id={args.campaign_id}")

    # Real campaign window length (in days)
    real_win = gt.filter(pl.col("in_campaign") == 1).select(
        pl.col("date").min().alias("start"),
        pl.col("date").max().alias("end"),
    )
    real_start = real_win["start"][0]
    real_end = real_win["end"][0]
    if real_start is None or real_end is None:
        raise ValueError("Could not infer real campaign window (in_campaign==1 missing).")

    win_len_days = int((real_end - real_start).days) + 1

    # Candidate placebo starts: must allow min_pre_days before placebo, and must end BEFORE real_start
    all_dates = gt.select(pl.col("date").unique()).sort("date")["date"].to_list()
    min_date = all_dates[0]

    latest_placebo_start = real_start - timedelta(days=win_len_days)
    earliest_placebo_start = min_date + timedelta(days=args.min_pre_days)

    candidates = [d for d in all_dates if (d >= earliest_placebo_start) and (d <= latest_placebo_start)]
    if len(candidates) == 0:
        raise ValueError(
            f"No placebo candidate starts found. "
            f"Try lowering --min_pre_days (currently {args.min_pre_days}) "
            f"or simulate more pre-period."
        )

    rng = np.random.default_rng(args.seed)
    n = min(args.n_placebos, len(candidates))
    placebo_starts_np = rng.choice(
    np.array(candidates, dtype="datetime64[D]"),
    size=n,
    replace=False,
)

    # numpy datetime64[D] -> python date
    placebo_starts = [dt_date.fromisoformat(str(d)) for d in placebo_starts_np]


    # Controls donor pool
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
    donor_wide = (
        donors.pivot(values="y", index="date", on="donor_id", aggregate_function="first")
        .sort("date")
        .fill_null(0)
    )

    # Treated aggregate time series (sum over treated units)
    treated_ts_base = (
        gt.filter(pl.col("treated") == 1)
        .group_by("date")
        .agg(pl.col("y_obs").sum().alias("y_treated"))
        .sort("date")
    )

    panel_base = (
        treated_ts_base.join(donor_wide, on="date", how="left")
        .fill_null(0)
        .sort("date")
    )

    donor_cols = [c for c in panel_base.columns if c not in ("date", "y_treated")]

    y_all_units = panel_base["y_treated"].to_numpy().astype(np.float64)
    X_all_units = panel_base.select(donor_cols).to_numpy().astype(np.float64)

    if args.use_log1p:
        y_all_fit = np.log1p(y_all_units)
        X_all_fit = np.log1p(X_all_units)
        fit_scale = "log1p"
    else:
        y_all_fit = y_all_units
        X_all_fit = X_all_units
        fit_scale = "units"

    rows = []
    method = f"scm_ridge_{args.donor_grain}" + ("_log1p" if args.use_log1p else "")

    for ps in placebo_starts:
        pe = ps + timedelta(days=win_len_days - 1)

        # pre-period = strictly before placebo start
        dates_np = panel_base["date"].to_numpy()
        pre_mask = dates_np < np.datetime64(ps)

        # placebo window mask
        plc_mask = (dates_np >= np.datetime64(ps)) & (dates_np <= np.datetime64(pe))

        # safety: must have some pre + some placebo points
        if pre_mask.sum() < 5 or plc_mask.sum() < 3:
            continue

        y_pre = y_all_fit[pre_mask]
        X_pre = X_all_fit[pre_mask, :]

        w = _ridge_weights(X_pre, y_pre, alpha=args.alpha)

        # predict and invert if log1p
        y0_hat_fit = X_all_fit @ w
        y0_hat = np.expm1(y0_hat_fit) if args.use_log1p else y0_hat_fit

        lift_units = y_all_units - y0_hat
        att_hat_units = float(np.mean(lift_units[plc_mask]))

        rmse_pre = float(np.sqrt(np.mean((y_pre - (X_pre @ w)) ** 2)))

        # stability CV on units in placebo window
        plc_lift = lift_units[plc_mask]
        mu = float(np.mean(plc_lift)) if plc_lift.size else np.nan
        sd = float(np.std(plc_lift, ddof=1)) if plc_lift.size > 1 else np.nan
        cv = (sd / abs(mu)) if (np.isfinite(sd) and np.isfinite(mu) and abs(mu) > 1e-9) else None

        rows.append(
            {
                "campaign_id": args.campaign_id,
                "method": method,
                "donor_grain": args.donor_grain,
                "use_log1p": bool(args.use_log1p),
                "alpha": float(args.alpha),
                "fit_scale": fit_scale,
                "placebo_start": ps,
                "placebo_end": pe,
                "att_hat_units": att_hat_units,
                "rmse_pre": rmse_pre,
                "cv": cv,
                "n_pre_days": int(pre_mask.sum()),
                "n_placebo_days": int(plc_mask.sum()),
            }
        )

    out = pl.DataFrame(rows)

    out_name = (
        f"scm_placebo_{args.campaign_id}_{args.donor_grain}"
        + ("_log1p" if args.use_log1p else "")
        + ".parquet"
    )
    out_path = processed / out_name
    out.write_parquet(out_path)

    print(f"Wrote placebo results: {out_path} ({out.height} rows)")


if __name__ == "__main__":
    main()
