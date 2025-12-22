from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def compute_truth(gt: pl.DataFrame) -> pl.DataFrame:
    """
    Compute multiple ground-truth ATT definitions from fact_ground_truth:

    - att_true_unit: mean(tau) for treated unit-days in campaign (matches DiD coefficient scale)
    - att_true_agg: mean over dates of sum(tau) across treated units (matches SCM aggregate-units scale)
    - att_true_log1p: mean over dates of [log1p(Y_treated) - log1p(Y_cf)] in campaign
      where Y_cf = sum(y_obs - tau) across treated units (counterfactual with no treatment)
    - att_true_pct: exp(att_true_log1p)-1
    """
    # unit-level truth
    att_true_unit = (
        gt.filter((pl.col("treated") == 1) & (pl.col("in_campaign") == 1))
          .select(pl.col("tau").mean().alias("att_true_unit"))
    )

    # daily treated aggregates (observed and counterfactual)
    daily = (
        gt.filter(pl.col("treated") == 1)
          .group_by("date")
          .agg(
              pl.col("in_campaign").max().alias("in_campaign"),
              pl.col("tau").sum().alias("tau_sum"),
              pl.col("y_obs").sum().alias("y_treated_sum"),
              (pl.col("y_obs") - pl.col("tau")).sum().alias("y_cf_sum"),
          )
          .sort("date")
    )

    att_true_agg = (
        daily.filter(pl.col("in_campaign") == 1)
             .select(pl.col("tau_sum").mean().alias("att_true_agg"))
    )

    # log1p truth: log-lift on treated aggregate series
    att_true_log = (
        daily.filter(pl.col("in_campaign") == 1)
             .with_columns(
                 (pl.col("y_treated_sum").log1p() - pl.col("y_cf_sum").log1p()).alias("log_lift")
             )
             .select(pl.col("log_lift").mean().alias("att_true_log1p"))
    )

    out = pl.concat([att_true_unit, att_true_agg, att_true_log], how="horizontal").with_columns((pl.col("att_true_log1p").exp() - 1.0).alias("att_true_pct"))
    return out


def choose_scale(method: str) -> str:
    # DiD uses unit ATT; SCM log1p uses log scale; SCM raw uses aggregate units
    if method == "twfe_did":
        return "unit_att"
    if "log1p" in method:
        return "log1p_att"
    if method.startswith("scm_"):
        return "agg_units_att"
    return "unit_att"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--campaign_id", type=str, default=None, help="If omitted, evaluates all campaigns found in results.")
    args = p.parse_args()

    processed = Path(args.processed_dir)
    results_path = processed / "fact_method_results.parquet"
    gt_path = processed / "fact_ground_truth.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}")

    results = pl.read_parquet(results_path)
    gt = pl.read_parquet(gt_path)

    if args.campaign_id:
        results = results.filter(pl.col("campaign_id") == args.campaign_id)
        gt = gt.filter(pl.col("campaign_id") == args.campaign_id)

    campaigns = sorted(results["campaign_id"].unique().to_list())

    rows = []
    for cid in campaigns:
        gt_c = gt.filter(pl.col("campaign_id") == cid)
        truth = compute_truth(gt_c)
        t_unit = float(truth["att_true_unit"][0])
        t_agg = float(truth["att_true_agg"][0])
        t_log = float(truth["att_true_log1p"][0])
        t_pct = float(truth["att_true_pct"][0])

        res_c = results.filter(pl.col("campaign_id") == cid)

        for r in res_c.iter_rows(named=True):
            method = r.get("method")
            scale = choose_scale(method)

            # pick method estimate to evaluate
            if scale == "log1p_att" and r.get("att_hat_fit") is not None:
                att_hat_used = float(r["att_hat_fit"])
                att_true_used = t_log
                truth_label = "log1p_att"
            elif scale == "agg_units_att":
                att_hat_used = float(r["att_hat"])
                att_true_used = t_agg
                truth_label = "agg_units_att"
            else:
                att_hat_used = float(r["att_hat"])
                att_true_used = t_unit
                truth_label = "unit_att"

            bias = att_hat_used - att_true_used
            rel_bias = bias / att_true_used if att_true_used != 0 else None

            rows.append({
                "campaign_id": cid,
                "method": method,
                "truth_used": truth_label,
                "att_hat_used": att_hat_used,
                "att_true_used": att_true_used,
                "bias": bias,
                "rel_bias": rel_bias,
                # include diagnostics if present
                "rmse_pre": r.get("rmse_pre"),
                "pretrend_p": r.get("pretrend_p"),
                # truth summary (handy for dashboard)
                "att_true_unit": t_unit,
                "att_true_agg": t_agg,
                "att_true_log1p": t_log,
                "att_true_pct": t_pct,
                # if method row has pct, keep it
                "att_hat_pct": r.get("att_hat_pct"),
            })

    out = pl.DataFrame(rows).sort(["campaign_id", "method"])
    out_path = processed / "fact_method_eval.parquet"
    out.write_parquet(out_path)
    print(out)


if __name__ == "__main__":
    main()
