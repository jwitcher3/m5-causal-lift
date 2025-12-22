from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import polars as pl


def _factorize(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return integer codes 0..G-1 and number of groups."""
    _, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int64), int(codes.max()) + 1


def _group_mean(x: np.ndarray, g: np.ndarray, G: int) -> np.ndarray:
    """Mean(x) within group for each observation."""
    denom = np.bincount(g, minlength=G).astype(np.float64)
    numer = np.bincount(g, weights=x, minlength=G).astype(np.float64)
    means = np.zeros(G, dtype=np.float64)
    mask = denom > 0
    means[mask] = numer[mask] / denom[mask]
    return means[g]


def _twoway_demean(x: np.ndarray, unit: np.ndarray, time: np.ndarray, iters: int = 10) -> np.ndarray:
    """
    Iterative demeaning for unbalanced panels:
      x <- x - E[x|unit] - E[x|time]  (alternating projections)
    """
    u_codes, U = _factorize(unit)
    t_codes, T = _factorize(time)

    r = x.astype(np.float64).copy()
    for _ in range(iters):
        r -= _group_mean(r, u_codes, U)
        r -= _group_mean(r, t_codes, T)

    # Center overall (harmless, can improve numerical stability)
    r -= r.mean()
    return r


def _cluster_se_1d(x: np.ndarray, e: np.ndarray, unit: np.ndarray) -> float:
    """One-way cluster-robust SE for single regressor after FE removal."""
    u_codes, U = _factorize(unit)
    xx = float((x * x).sum())
    if xx <= 0:
        return float("nan")

    xe_sum = np.bincount(u_codes, weights=x * e, minlength=U).astype(np.float64)
    meat = float((xe_sum * xe_sum).sum())

    # Finite-sample correction
    n = x.shape[0]
    k = 1
    if U > 1:
        scale = (U / (U - 1)) * ((n - 1) / (n - k))
    else:
        scale = 1.0

    var = scale * meat / (xx * xx)
    return float(np.sqrt(var))


def load_panel(gt_path: Path, grain: str, campaign_id: str) -> pl.DataFrame:
    keys = ["store_id", "dept_id"] if grain == "store_dept" else ["store_id", "item_id"]
    df = pl.read_parquet(gt_path)

    if df["date"].dtype != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))

    df = df.filter(pl.col("campaign_id") == campaign_id).with_columns(
        pl.concat_str([pl.col(k).cast(pl.Utf8) for k in keys], separator="|").alias("unit_id"),
        pl.col("date").cast(pl.Utf8).alias("date_id"),  # stable for factorization
        (pl.col("in_campaign") == 1).cast(pl.Int8).alias("post"),
        (pl.col("treated") == 1).cast(pl.Int8).alias("treated"),
    )

    return df


def true_att(df: pl.DataFrame) -> float:
    return float(
        df.lazy()
        .filter((pl.col("treated") == 1) & (pl.col("in_campaign") == 1))
        .select(pl.col("tau").mean())
        .collect()
        .item()
    )


def twfe_did(df: pl.DataFrame) -> dict:
    """
    TWFE DiD via iterative demeaning, then OLS on x = treated*post.
    """
    d = df.select(["unit_id", "date_id", "y_obs", "treated", "post"]).to_pandas()

    unit = d["unit_id"].to_numpy()
    time = d["date_id"].to_numpy()
    y = d["y_obs"].to_numpy(dtype=np.float64)
    x = (d["treated"].to_numpy(dtype=np.float64) * d["post"].to_numpy(dtype=np.float64))

    y_t = _twoway_demean(y, unit, time, iters=10)
    x_t = _twoway_demean(x, unit, time, iters=10)

    denom = float((x_t * x_t).sum())
    if denom <= 0:
        return {"att_hat": float("nan"), "se": float("nan"), "n": int(len(y))}

    beta = float((x_t * y_t).sum() / denom)
    e = y_t - beta * x_t

    se = _cluster_se_1d(x_t, e, unit)
    return {"att_hat": beta, "se": se, "n": int(len(y))}


def event_study(df: pl.DataFrame, min_lead: int = -14, max_lag: int = 28, omit: int = -1) -> pl.DataFrame:
    """
    Event study via FE removal + OLS on indicators 1[rel_day=k]*treated for k in window excluding omit.
    Cluster-robust SE by unit.
    """
    d = df.select(["unit_id", "date_id", "y_obs", "treated", "rel_day"]).to_pandas()

    unit = d["unit_id"].to_numpy()
    time = d["date_id"].to_numpy()
    y = d["y_obs"].to_numpy(dtype=np.float64)
    treated = d["treated"].to_numpy(dtype=np.float64)
    rel = d["rel_day"].to_numpy(dtype=np.int64)

    mask = (rel >= min_lead) & (rel <= max_lag)
    unit = unit[mask]
    time = time[mask]
    y = y[mask]
    treated = treated[mask]
    rel = rel[mask]

    ks = [k for k in range(min_lead, max_lag + 1) if k != omit]

    # Build design matrix (n x K)
    X = np.column_stack([(treated * (rel == k).astype(np.float64)) for k in ks])

    # Residualize y and each column of X w.r.t unit/time FEs
    y_t = _twoway_demean(y, unit, time, iters=10)
    X_t = np.column_stack([_twoway_demean(X[:, j], unit, time, iters=10) for j in range(X.shape[1])])

    # OLS
    beta, *_ = np.linalg.lstsq(X_t, y_t, rcond=None)
    e = y_t - X_t @ beta

    # Cluster robust variance (one-way by unit)
    u_codes, U = _factorize(unit)
    XtX = X_t.T @ X_t
    try:
        bread = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtX)

    meat = np.zeros((X_t.shape[1], X_t.shape[1]), dtype=np.float64)
    for g in range(U):
        idx = (u_codes == g)
        if not np.any(idx):
            continue
        Xg = X_t[idx, :]
        eg = e[idx]
        s = Xg.T @ eg  # K-vector
        meat += np.outer(s, s)

    n = X_t.shape[0]
    k = X_t.shape[1]
    scale = (U / (U - 1)) * ((n - 1) / (n - k)) if U > 1 else 1.0
    V = scale * (bread @ meat @ bread)

    se = np.sqrt(np.clip(np.diag(V), 0, None))

    out = pl.DataFrame(
        {
            "rel_day": ks,
            "beta": beta.astype(float).tolist(),
            "se": se.astype(float).tolist(),
        }
    ).with_columns(
        (pl.col("beta") - 1.96 * pl.col("se")).alias("ci_lo"),
        (pl.col("beta") + 1.96 * pl.col("se")).alias("ci_hi"),
    ).sort("rel_day")

    return out, V, ks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, default="data/processed")
    p.add_argument("--grain", type=str, default="store_dept", choices=["store_dept", "store_item"])
    p.add_argument("--campaign_id", type=str, default="cmp_001")
    args = p.parse_args()

    processed = Path(args.processed_dir)
    gt_path = processed / "fact_ground_truth.parquet"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}. Run simulator first.")

    df = load_panel(gt_path, grain=args.grain, campaign_id=args.campaign_id)

    att_true = true_att(df)
    did = twfe_did(df)
    
es, V, ks = event_study(df)

    # Joint pretrend test on leads (-14..-2)
    from scipy.stats import chi2
    lead_idx = [i for i,k in enumerate(ks) if (-14 <= k <= -2)]
    if len(lead_idx) > 0:
        b = es.select("beta").to_numpy().reshape(-1)
        bL = b[lead_idx]
        VL = V[np.ix_(lead_idx, lead_idx)]
        try:
            VL_inv = np.linalg.inv(VL)
        except np.linalg.LinAlgError:
            VL_inv = np.linalg.pinv(VL)
        chi2_stat = float(bL.T @ VL_inv @ bL)
        df_chi2 = int(len(lead_idx))
        pretrend_p = float(1.0 - chi2.cdf(chi2_stat, df_chi2))
    else:
        chi2_stat, df_chi2, pretrend_p = float("nan"), 0, float("nan")

    results = pl.DataFrame(
        [{
            "campaign_id": args.campaign_id,
            "method": "twfe_did",
            "att_hat": did["att_hat"],
            "att_se": did["se"],
            "att_true": att_true,
            "bias": did["att_hat"] - att_true,
            "n": did["n"],
            "pretrend_chi2": chi2_stat,
            "pretrend_df": df_chi2,
            "pretrend_p": pretrend_p,
        }]
    )

    out_results = processed / "fact_method_results.parquet"
    if out_results.exists():
        prev = pl.read_parquet(out_results)
        results = pl.concat([prev, results], how="vertical").unique(subset=["campaign_id", "method"], keep="last")

    results.write_parquet(out_results)
    es.write_parquet(processed / f"event_study_{args.campaign_id}.parquet")

    print(results)


    if __name__ == "__main__":
        main()
