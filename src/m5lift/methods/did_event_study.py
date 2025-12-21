from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def _factorize(arr: np.ndarray) -> tuple[np.ndarray, int]:
    _, codes = np.unique(arr, return_inverse=True)
    codes = codes.astype(np.int64)
    return codes, int(codes.max()) + 1


def _group_mean(x: np.ndarray, g: np.ndarray, G: int) -> np.ndarray:
    denom = np.bincount(g, minlength=G).astype(np.float64)
    numer = np.bincount(g, weights=x, minlength=G).astype(np.float64)
    means = np.zeros(G, dtype=np.float64)
    mask = denom > 0
    means[mask] = numer[mask] / denom[mask]
    return means[g]


def _twoway_demean(x: np.ndarray, unit: np.ndarray, time: np.ndarray, iters: int = 10) -> np.ndarray:
    u_codes, U = _factorize(unit)
    t_codes, T = _factorize(time)

    r = x.astype(np.float64).copy()
    for _ in range(iters):
        r -= _group_mean(r, u_codes, U)
        r -= _group_mean(r, t_codes, T)

    r -= r.mean()
    return r


def _cluster_vcov(X: np.ndarray, e: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """
    One-way cluster-robust variance for OLS coefficients.
    X: (n,k) already residualized for FEs
    e: (n,) residuals
    """
    u_codes, U = _factorize(unit)

    XtX = X.T @ X
    try:
        bread = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtX)

    meat = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    for g in range(U):
        idx = (u_codes == g)
        if not np.any(idx):
            continue
        Xg = X[idx, :]
        eg = e[idx]
        s = Xg.T @ eg
        meat += np.outer(s, s)

    n = X.shape[0]
    k = X.shape[1]
    scale = (U / (U - 1)) * ((n - 1) / (n - k)) if U > 1 and (n - k) > 0 else 1.0
    return scale * (bread @ meat @ bread)


def load_panel(gt_path: Path, grain: str, campaign_id: str) -> pl.DataFrame:
    keys = ["store_id", "dept_id"] if grain == "store_dept" else ["store_id", "item_id"]
    df = pl.read_parquet(gt_path)

    if df["date"].dtype != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))

    df = df.filter(pl.col("campaign_id") == campaign_id).with_columns(
        pl.concat_str([pl.col(k).cast(pl.Utf8) for k in keys], separator="|").alias("unit_id"),
        pl.col("date").cast(pl.Utf8).alias("date_id"),
        (pl.col("treated") == 1).cast(pl.Int8).alias("treated"),
        (pl.col("in_campaign") == 1).cast(pl.Int8).alias("post"),
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
    unit = df["unit_id"].to_numpy()
    time = df["date_id"].to_numpy()
    y = df["y_obs"].to_numpy().astype(np.float64)
    treated = df["treated"].to_numpy().astype(np.float64)
    post = df["post"].to_numpy().astype(np.float64)

    x = treated * post

    y_t = _twoway_demean(y, unit, time, iters=10)
    x_t = _twoway_demean(x, unit, time, iters=10)

    denom = float((x_t * x_t).sum())
    if denom <= 0:
        return {"att_hat": float("nan"), "se": float("nan"), "n": int(len(y))}

    beta = float((x_t * y_t).sum() / denom)
    e = y_t - beta * x_t

    X = x_t.reshape(-1, 1)
    V = _cluster_vcov(X, e, unit)
    se = float(np.sqrt(max(V[0, 0], 0.0)))

    return {"att_hat": beta, "se": se, "n": int(len(y))}


def event_study(df: pl.DataFrame, min_lead: int = -14, max_lag: int = 28, omit: int = -1):
    """
    Event study with unit+date FE removed.
    IMPORTANT: keep omit *rows* in the sample; omit only the indicator column (reference period).
    """
    unit = df["unit_id"].to_numpy()
    time = df["date_id"].to_numpy()
    y = df["y_obs"].to_numpy().astype(np.float64)
    treated = df["treated"].to_numpy().astype(np.float64)
    rel = df["rel_day"].to_numpy().astype(np.int64)

    # Keep full window rows (do NOT drop omit rows)
    mask = (rel >= min_lead) & (rel <= max_lag)
    unit = unit[mask]
    time = time[mask]
    y = y[mask]
    treated = treated[mask]
    rel = rel[mask]

    ks = [k for k in range(min_lead, max_lag + 1) if k != omit]

    # X columns: treated * 1[rel_day == k]
    X = np.column_stack([(treated * (rel == k).astype(np.float64)) for k in ks])

    # Residualize
    y_t = _twoway_demean(y, unit, time, iters=10)
    X_t = np.column_stack([_twoway_demean(X[:, j], unit, time, iters=10) for j in range(X.shape[1])])

    # OLS
    beta, *_ = np.linalg.lstsq(X_t, y_t, rcond=None)
    e = y_t - X_t @ beta

    V = _cluster_vcov(X_t, e, unit)
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


def _align_columns(prev: pl.DataFrame, cur: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Align schemas for concat, casting NULL columns to target dtypes."""
    prev_schema = prev.schema
    cur_schema = cur.schema

    all_cols = sorted(set(prev_schema.keys()) | set(cur_schema.keys()))

    # Target dtype preference: current run (cur) wins; else fallback to prev
    target_dtype = {c: (cur_schema.get(c) or prev_schema.get(c) or pl.Null) for c in all_cols}

    def coerce(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
        for c in all_cols:
            dt = target_dtype[c]
            if c not in schema:
                # create a typed null column
                df = df.with_columns(pl.lit(None).cast(dt).alias(c))
            else:
                # cast null-typed or mismatched columns to target dtype
                if schema[c] == pl.Null and dt != pl.Null:
                    df = df.with_columns(pl.col(c).cast(dt))
                elif schema[c] != dt and dt != pl.Null:
                    df = df.with_columns(pl.col(c).cast(dt))
        return df.select(all_cols)

    prev2 = coerce(prev, prev_schema)
    cur2 = coerce(cur, cur_schema)
    return prev2, cur2


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

    # Simple descriptive (don’t use as a “test”)
    mean_lead_beta = float(
        es.filter((pl.col("rel_day") <= -2) & (pl.col("rel_day") >= -14))
          .select(pl.col("beta").mean())
          .item()
    )

    # Proper joint pretrend test on leads (-14..-2): H0: all lead betas = 0
    try:
        from scipy.stats import chi2
        lead_idx = [i for i, k in enumerate(ks) if (-14 <= k <= -2)]
        if lead_idx:
            b = np.array(es["beta"], dtype=np.float64)
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
    except Exception:
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
            "mean_lead_beta": mean_lead_beta,
            "pretrend_chi2": chi2_stat,
            "pretrend_df": df_chi2,
            "pretrend_p": pretrend_p,
        }]
    )

    out_results = processed / "fact_method_results.parquet"
    if out_results.exists():
        prev = pl.read_parquet(out_results)
        prev, results = _align_columns(prev, results)
        results = pl.concat([prev, results], how="vertical").unique(subset=["campaign_id", "method"], keep="last")

    results.write_parquet(out_results)
    es.write_parquet(processed / f"event_study_{args.campaign_id}.parquet")

    print(results)


if __name__ == "__main__":
    main()
