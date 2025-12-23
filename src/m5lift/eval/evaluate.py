from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import polars as pl


def compute_truth(gt_c: pl.DataFrame) -> pl.DataFrame:
    treated = pl.col("treated").cast(pl.Int8) == 1
    in_camp = pl.col("in_campaign").cast(pl.Int8) == 1

    # Unit-day ATT: mean tau across treated unit-days during campaign
    att_true_unit = (
        gt_c.filter(treated & in_camp)
        .select(pl.col("tau").mean().alias("att_true_unit"))
    )

    # Aggregate treated-series ATT (units): sum tau by date then average across campaign dates
    treated_daily = (
        gt_c.filter(treated)
        .group_by("date")
        .agg(
            pl.col("tau").sum().alias("tau_sum"),
            pl.col("y_obs").sum().alias("y_obs_sum"),
            (pl.col("y_obs") - pl.col("tau")).sum().alias("y0_sum"),
            pl.col("in_campaign").max().cast(pl.Int8).alias("in_campaign"),
        )
        .sort("date")
    )

    att_true_agg = (
        treated_daily.filter(pl.col("in_campaign") == 1)
        .select(pl.col("tau_sum").mean().alias("att_true_agg"))
    )

    # Log1p truth on aggregate series: mean of log1p(y) - log1p(y0) over campaign dates
    att_true_log1p = (
        treated_daily.filter(pl.col("in_campaign") == 1)
        .with_columns((pl.col("y_obs_sum").log1p() - pl.col("y0_sum").log1p()).alias("dlog1p"))
        .select(pl.col("dlog1p").mean().alias("att_true_log1p"))
    )

    out = att_true_unit.join(att_true_agg, how="cross").join(att_true_log1p, how="cross")

    # Percent lift implied by log1p ATT
    out = out.with_columns(
        pl.col("att_true_log1p")
        .map_elements(
            lambda x: float(np.expm1(x)) if x is not None else None,
            return_dtype=pl.Float64,
        )
        .alias("att_true_pct")
    )

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

    rows: list[dict] = []
    for cid in campaigns:
        gt_c = gt.filter(pl.col("campaign_id") == cid)
        truth = compute_truth(gt_c)

        t_unit = truth.select("att_true_unit").item()
        t_agg = truth.select("att_true_agg").item()
        t_log = truth.select("att_true_log1p").item()
        t_pct = truth.select("att_true_pct").item()

        if t_unit is None or t_agg is None or t_log is None:
            raise ValueError(f"Truth contains nulls for campaign_id={cid}: {truth}")

        res_c = results.filter(pl.col("campaign_id") == cid)

        for r in res_c.iter_rows(named=True):
            method = r.get("method") or ""
            scale = choose_scale(method)

            # pick method estimate to evaluate
            if scale == "log1p_att" and r.get("att_hat_fit") is not None:
                att_hat_used = float(r["att_hat_fit"])
                att_true_used = float(t_log)
                truth_label = "log1p_att"
            elif scale == "agg_units_att":
                att_hat_used = float(r["att_hat"])
                att_true_used = float(t_agg)
                truth_label = "agg_units_att"
            else:
                att_hat_used = float(r["att_hat"])
                att_true_used = float(t_unit)
                truth_label = "unit_att"

            bias = att_hat_used - att_true_used
            rel_bias = (bias / att_true_used) if att_true_used != 0 else None

            rows.append({
                "campaign_id": cid,
                "method": method,
                "truth_used": truth_label,
                "att_hat_used": att_hat_used,
                "att_true_used": att_true_used,
                "bias": bias,
                "rel_bias": rel_bias,
                # diagnostics if present
                "rmse_pre": r.get("rmse_pre"),
                "pretrend_p": r.get("pretrend_p"),
                # truth summary
                "att_true_unit": t_unit,
                "att_true_agg": t_agg,
                "att_true_log1p": t_log,
                "att_true_pct": t_pct,
                # keep method pct if present
                "att_hat_pct": r.get("att_hat_pct"),
            })

    if not rows:
        raise ValueError("No evaluation rows produced (empty results?).")

    out = pl.DataFrame(rows).sort(["campaign_id", "method"])
    out_path = processed / "fact_method_eval.parquet"
    out.write_parquet(out_path)
    print(out)


if __name__ == "__main__":
    main()
