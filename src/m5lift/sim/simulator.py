from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import polars as pl


@dataclass
class CampaignSpec:
    campaign_id: str
    name: str
    start_date: str          # "YYYY-MM-DD"
    end_date: str            # "YYYY-MM-DD"
    treat_frac: float = 0.20
    max_uplift: float = 0.15 # 15% peak lift
    ramp_days: int = 3
    decay_days: int = 5
    seed: int = 7


def _lift_multiplier(rel_day: pl.Expr, total_days: int, max_uplift: float, ramp_days: int, decay_days: int) -> pl.Expr:
    """
    Piecewise lift: ramp up -> plateau -> decay
    Returns multiplier (1 + uplift), where uplift in [0, max_uplift].
    """
    # Ramp: 0..ramp_days-1
    ramp = (rel_day.cast(pl.Float64) / pl.lit(max(ramp_days, 1))).clip(0, 1) * max_uplift

    # Plateau: after ramp until decay window starts
    plateau_uplift = pl.lit(max_uplift)

    # Decay: last decay_days of campaign
    # decay_progress: 0 at start of decay window -> 1 at end
    decay_start = pl.lit(max(total_days - decay_days, 0))
    decay_progress = ((rel_day - decay_start).cast(pl.Float64) / pl.lit(max(decay_days, 1))).clip(0, 1)
    decay = (pl.lit(max_uplift) * (pl.lit(1.0) - decay_progress)).clip(0, max_uplift)

    uplift = (
        pl.when(rel_day < 0).then(0.0)
        .when(rel_day < ramp_days).then(ramp)
        .when(rel_day < (total_days - decay_days)).then(plateau_uplift)
        .when(rel_day <= (total_days - 1)).then(decay)
        .otherwise(0.0)
    )

    return pl.lit(1.0) + uplift


def simulate_campaign(
    features_path: Path,
    out_dir: Path,
    grain: str,
    spec: CampaignSpec,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keys by grain
    keys = ["store_id", "dept_id"] if grain == "store_dept" else ["store_id", "item_id"]

    df = pl.read_parquet(features_path)

    # Ensure date type
    if df["date"].dtype != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))

    # Define mu0 as a simple baseline: roll_mean_28 fallback to roll_mean_7 then lag_7 then 0
    df = df.with_columns(
        pl.coalesce(
            pl.col("roll_mean_28"),
            pl.col("roll_mean_7"),
            pl.col("lag_7"),
            pl.lit(0.0),
        ).cast(pl.Float64).alias("mu0")
    )

    # Candidate units are those with enough history (mu0 not null and >0 a bit)
    unit_df = (
        df.group_by(keys)
        .agg(
            pl.col("mu0").mean().alias("mu0_mean"),
            pl.len().alias("n_rows"),
        )
        .filter(pl.col("n_rows") > 60)  # enough days for pre/post
        .sort("mu0_mean", descending=True)
    )

    # Sample treated units
    n_units = unit_df.height
    n_treated = max(1, int(np.floor(spec.treat_frac * n_units)))

    rng = np.random.default_rng(spec.seed)
    treated_idx = rng.choice(n_units, size=n_treated, replace=False)

    treated_units = unit_df.select(keys)[treated_idx]
    treated_units = treated_units.with_columns(pl.lit(1).alias("treated_unit"))


    # Join treated flag back to daily panel
    df = df.join(treated_units, on=keys, how="left").with_columns(
        pl.col("treated_unit").fill_null(0).cast(pl.Int8).alias("treated")
    )

    # Campaign window + relative day
    start = pl.lit(spec.start_date).str.strptime(pl.Date, "%Y-%m-%d")
    end = pl.lit(spec.end_date).str.strptime(pl.Date, "%Y-%m-%d")
    df = df.with_columns(
        [
            (pl.col("date") - start).dt.total_days().alias("rel_day"),
            ((pl.col("date") >= start) & (pl.col("date") <= end)).cast(pl.Int8).alias("in_campaign"),
        ]
    )

    # Total days inclusive (compute in Python for Polars compatibility)
    from datetime import date

    y1, m1, d1 = map(int, spec.start_date.split("-"))
    y2, m2, d2 = map(int, spec.end_date.split("-"))
    total_days = (date(y2, m2, d2) - date(y1, m1, d1)).days + 1

    # Treatment effect tau only when treated & in_campaign
    mult = _lift_multiplier(pl.col("rel_day"), total_days, spec.max_uplift, spec.ramp_days, spec.decay_days)
    df = df.with_columns(
        [
            pl.when((pl.col("treated") == 1) & (pl.col("in_campaign") == 1))
            .then(pl.col("mu0") * (mult - 1.0))
            .otherwise(0.0)
            .alias("tau")
        ]
    )

    # Observed outcome y_obs: apply lift on mu0 + noise
    # Noise: Gaussian on log(1+mu), then back-transform; keeps positives and scale-ish reasonable.
    # y_obs = round(exp(log1p(mu0+tau) + eps) - 1)
    eps = rng.normal(0.0, 0.35, size=df.height)  # tunable noise
    df = df.with_columns(pl.Series("eps", eps).cast(pl.Float64))

    df = df.with_columns(
        [
            (pl.col("mu0") + pl.col("tau")).clip(0, None).alias("mu1"),
            (pl.col("mu0")).clip(0, None).alias("mu0_clip"),
        ]
    ).with_columns(
        [
            (pl.col("mu1").add(1.0).log() + pl.col("eps")).alias("log_y_obs"),
            (pl.col("mu0_clip").add(1.0).log() + pl.col("eps")).alias("log_y0_draw"),
        ]
    ).with_columns(
        [
            (pl.col("log_y_obs").exp() - 1.0).round(0).clip(0, None).cast(pl.Int64).alias("y_obs"),
            (pl.col("log_y0_draw").exp() - 1.0).round(0).clip(0, None).cast(pl.Int64).alias("y0_draw"),
        ]
    )

    # Write outputs
    dim_campaign = pl.DataFrame(
        {
            "campaign_id": [spec.campaign_id],
            "name": [spec.name],
            "start_date": [spec.start_date],
            "end_date": [spec.end_date],
            "treat_frac": [spec.treat_frac],
            "max_uplift": [spec.max_uplift],
            "ramp_days": [spec.ramp_days],
            "decay_days": [spec.decay_days],
            "seed": [spec.seed],
        }
    )
    dim_campaign.write_parquet(out_dir / "dim_campaign.parquet")

    # Treatment panel (daily treated indicator)
    treat_panel = df.select(["campaign_id"] if "campaign_id" in df.columns else [])  # no-op
    treat_panel = df.select(keys + ["date", "treated", "in_campaign", "rel_day"]).with_columns(
        pl.lit(spec.campaign_id).alias("campaign_id"),
        pl.lit(1.0).alias("intensity"),
    )
    treat_panel.write_parquet(out_dir / "fact_treatment_panel.parquet")

    # Ground truth panel
    gt = df.select(
        keys
        + ["date", "mu0", "tau", "y_obs", "y0_draw", "treated", "in_campaign", "rel_day"]
    ).with_columns(
        pl.lit(spec.campaign_id).alias("campaign_id")
    )
    gt.write_parquet(out_dir / "fact_ground_truth.parquet")

    # Print quick summary
    post_mask = (gt["date"] >= pl.date(spec.start_date)) & (gt["date"] <= pl.date(spec.end_date))
    treated_mask = gt["treated"] == 1
    att_true = gt.filter(post_mask & treated_mask).select(pl.mean("tau")).item()
    print(f"Campaign {spec.campaign_id}: treated_units={n_treated:,}/{n_units:,} true_ATT(mean tau in-campaign)={att_true:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--grain", type=str, default="store_dept", choices=["store_dept", "store_item"])
    p.add_argument("--campaign_id", type=str, default="cmp_001")
    p.add_argument("--name", type=str, default="Synthetic Campaign v1")
    p.add_argument("--start_date", type=str, default="2014-06-01")
    p.add_argument("--end_date", type=str, default="2014-06-28")
    p.add_argument("--treat_frac", type=float, default=0.20)
    p.add_argument("--max_uplift", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    processed = Path(args.processed_dir)
    features_path = processed / "feat_sales_lags.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Build lag features first.")

    spec = CampaignSpec(
        campaign_id=args.campaign_id,
        name=args.name,
        start_date=args.start_date,
        end_date=args.end_date,
        treat_frac=args.treat_frac,
        max_uplift=args.max_uplift,
        seed=args.seed,
    )

    simulate_campaign(
        features_path=features_path,
        out_dir=processed,
        grain=args.grain,
        spec=spec,
    )


if __name__ == "__main__":
    main()
