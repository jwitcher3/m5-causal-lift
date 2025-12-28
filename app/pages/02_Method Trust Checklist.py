from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Method Trust Checklist", layout="wide")
st.title("Method Trust Checklist")
st.caption("Guardrails + placebo tests to sanity-check whether an SCM lift looks real.")

DEFAULT_PROCESSED = Path("data/processed")
processed_dir = Path(st.sidebar.text_input("processed_dir", str(DEFAULT_PROCESSED)))

results_path = processed_dir / "fact_method_results.parquet"
eval_path = processed_dir / "fact_method_eval.parquet"

if not results_path.exists():
    st.error(f"Missing {results_path}. Run methods first.")
    st.stop()

results = pl.read_parquet(results_path)
campaigns = sorted(results["campaign_id"].unique().to_list())
campaign_id = st.sidebar.selectbox("campaign_id", campaigns, index=0)

# Try to pick "best" SCM from eval if available; else fallback to any scm_* result
best_method = None
grain = "store_dept"
log1p = True

if eval_path.exists():
    ev = pl.read_parquet(eval_path).filter(pl.col("campaign_id") == campaign_id)
    best_scm = (
        ev.filter(pl.col("method").str.contains("^scm_"))
          .filter(~pl.col("method").str.contains("simplex"))
          .filter(~pl.col("method").str.contains("log1p_sim"))
          .sort(pl.col("bias").abs())
    )
    if best_scm.height > 0:
        best_method = best_scm.row(0, named=True)["method"]

if best_method is None:
    # fallback: pick any scm row in results
    cand = results.filter(
        (pl.col("campaign_id") == campaign_id) & (pl.col("method").str.contains("^scm_"))
    ).sort("method")
    if cand.height > 0:
        best_method = cand.row(0, named=True)["method"]

if best_method is None:
    st.warning("No SCM results found for this campaign yet. Run: make scm ...")
    st.stop()

grain = "store_dept" if "store_dept" in best_method else "store"
log1p = ("log1p" in best_method)

# Pull SCM point estimate + fit metric from results table
scm_row = (
    results.filter((pl.col("campaign_id") == campaign_id) & (pl.col("method") == best_method))
)
att_hat_units = float(scm_row.select(pl.col("att_hat")).item()) if scm_row.height else None
rmse_pre = float(scm_row.select(pl.col("rmse_pre")).item()) if (scm_row.height and "rmse_pre" in scm_row.columns) else None

c1, c2, c3 = st.columns(3)
c1.metric("Selected SCM method", best_method)
c2.metric("ATT (units)", f"{att_hat_units:,.2f}" if att_hat_units is not None else "—")
c3.metric("Pre-fit RMSE", f"{rmse_pre:.4f}" if rmse_pre is not None else "—")

st.divider()

# --- Placebo file ---
suffix = "_log1p" if log1p else ""
placebo_path = processed_dir / f"scm_placebo_{campaign_id}_{grain}{suffix}.parquet"

st.subheader("Placebo test (fake treatment windows)")
st.caption("If the campaign lift is real, it should look extreme relative to placebo lifts.")

if not placebo_path.exists():
    st.warning(f"Missing placebo file: {placebo_path.name}")
    st.code(
        f"python src/m5lift/methods/placebo_scm.py --processed_dir {processed_dir} "
        f"--campaign_id {campaign_id} --donor_grain {grain} "
        f"{'--use_log1p ' if log1p else ''}--alpha 50 --n_placebos 50",
        language="bash",
    )
    st.stop()

pb = pl.read_parquet(placebo_path)
if "att_hat_units" not in pb.columns:
    st.error(f"Expected att_hat_units in placebo file, found: {pb.columns}")
    st.stop()

pb_lifts = pb["att_hat_units"].to_numpy().astype(float)

# empirical two-sided p-value: how often placebo is as extreme as actual
p_val = None
if att_hat_units is not None:
    p_val = float(np.mean(np.abs(pb_lifts) >= abs(att_hat_units)))

c1, c2, c3 = st.columns(3)
c1.metric("N placebos", f"{len(pb_lifts)}")
c2.metric("Placebo lift std", f"{np.std(pb_lifts):,.2f}")
c3.metric("Empirical p-value (2-sided)", f"{p_val:.3f}" if p_val is not None else "—")

# Histogram
fig, ax = plt.subplots()
ax.hist(pb_lifts, bins=20)
ax.set_title("Placebo lift distribution (units)")
ax.set_xlabel("ATT (units)")
ax.set_ylabel("Count")
if att_hat_units is not None:
    ax.axvline(att_hat_units, linewidth=2)
    ax.axvline(-att_hat_units, linewidth=2)
st.pyplot(fig)

# Show extremes
st.subheader("Most extreme placebo windows")
pb_show = (
    pb.with_columns(pl.col("att_hat_units").abs().alias("abs_lift"))
      .sort("abs_lift", descending=True)
      .select(["placebo_start","placebo_end","att_hat_units","rmse_pre","cv","n_pre_days","n_placebo_days"])
      .head(10)
)
st.dataframe(pb_show.to_pandas())

# Simple checklist
st.subheader("Checklist (SCM)")
checks = []

if rmse_pre is not None and rmse_pre > 0.35:
    checks.append(f"⚠️ Pre-fit RMSE is high: {rmse_pre:.3f} (tune alpha / donor grain / log1p).")
else:
    checks.append("✅ Pre-fit RMSE looks acceptable (or not available).")

if p_val is not None and p_val < 0.10:
    checks.append(f"✅ Placebo test passes (p={p_val:.3f}): lift is relatively rare under placebo.")
elif p_val is not None:
    checks.append(f"⚠️ Placebo test weak (p={p_val:.3f}): lift is common under placebo → be cautious.")
else:
    checks.append("⚠️ Missing p-value (no SCM ATT found).")

st.markdown("\n".join([f"- {c}" for c in checks]))
