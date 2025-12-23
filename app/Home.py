from __future__ import annotations

from pathlib import Path
import polars as pl
import streamlit as st
import numpy as np

st.set_page_config(page_title="M5 Causal Lift", layout="wide")

DEFAULT_PROCESSED = Path("data/processed")

st.title("M5 Causal Lift (Incrementality Sandbox)")
st.caption("DiD / Event Study + Synthetic Control evaluated against simulated ground truth (M5 base).")

processed_dir = Path(st.sidebar.text_input("processed_dir", str(DEFAULT_PROCESSED)))

results_path = processed_dir / "fact_method_results.parquet"
if not results_path.exists():
    st.error(f"Missing {results_path}. Run simulator + methods first.")
    st.stop()

results = pl.read_parquet(results_path)

campaigns = sorted(results["campaign_id"].unique().to_list())
campaign_id = st.sidebar.selectbox("campaign_id", campaigns, index=0)

# --- Eval table (scale-aware) ---
eval_path = processed_dir / "fact_method_eval.parquet"
ev = None
if eval_path.exists():
    ev = pl.read_parquet(eval_path).filter(pl.col("campaign_id") == campaign_id)
    ev = ev.with_columns(pl.col("bias").abs().alias("abs_bias"))

    st.subheader("Evaluation vs ground truth (scale-aware)")
    ev_show = ev.filter(~pl.col("method").str.contains("simplex")).sort("abs_bias")
    st.dataframe(ev_show.to_pandas(), use_container_width=True)
else:
    st.caption("No eval table found. Run: python src/m5lift/eval/evaluate.py")

# --- Choose best SCM method automatically (log1p preferred) ---
best_scm_method = None
best_scm_row = None
if ev is not None:
    best = (
        ev.filter(
            (pl.col("method").str.contains("^scm_"))
            & (pl.col("truth_used") == "log1p_att")
            & (~pl.col("method").str.contains("simplex"))
        )
        .sort("abs_bias")
    )
    if best.height > 0:
        best_scm_row = best.row(0, named=True)
        best_scm_method = best_scm_row["method"]

st.sidebar.subheader("SCM selection")
auto_pick = st.sidebar.checkbox("Auto-pick best SCM (from eval)", value=True)

# build options from files that might exist
series_candidates = []
for f in [
    f"scm_series_{campaign_id}_store.parquet",
    f"scm_series_{campaign_id}_store_dept.parquet",
    f"scm_series_{campaign_id}_store_dept_log1p.parquet",
]:
    if (processed_dir / f).exists():
        series_candidates.append(f)

# if auto-pick, map best method -> expected series filename
picked_series = None
if auto_pick and best_scm_method is not None:
    grain = "store_dept" if "store_dept" in best_scm_method else "store"
    log1p = "_log1p" if "log1p" in best_scm_method else ""
    expected = f"scm_series_{campaign_id}_{grain}{log1p}.parquet"
    if (processed_dir / expected).exists():
        picked_series = expected

if not series_candidates:
    st.warning("No SCM series parquet found. Run synth_control.py to generate one.")
    series_file = None
else:
    default_idx = 0
    if picked_series is not None and picked_series in series_candidates:
        default_idx = series_candidates.index(picked_series)

    series_file = st.sidebar.selectbox("SCM series file", series_candidates, index=default_idx)

# --- Plot SCM series + show lift metrics ---
if series_file:
    series_path = processed_dir / series_file
    ts = pl.read_parquet(series_path).sort("date")
    ts_pd = ts.to_pandas().set_index("date")

    st.subheader("Synthetic control: treated vs counterfactual")
    st.line_chart(ts_pd[["y_treated", "y0_hat"]], use_container_width=True)

    st.subheader("Synthetic control: estimated lift over time")
    st.line_chart(ts_pd[["lift_hat"]], use_container_width=True)

    # Metrics (true vs estimated) from eval row if available
    if best_scm_row is not None:
        true_pct = float(best_scm_row.get("att_true_pct"))
        est_pct = None

        # prefer att_hat_pct if present; else compute from att_hat_used on log scale
        if best_scm_row.get("att_hat_pct") is not None:
            est_pct = float(best_scm_row["att_hat_pct"])
        else:
            # If this row is log1p_att, att_hat_used is log-lift
            if best_scm_row.get("truth_used") == "log1p_att":
                est_pct = float(np.expm1(float(best_scm_row["att_hat_used"])))

        if est_pct is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("True lift (%), log1p truth", f"{true_pct*100:.2f}%")
            c2.metric("Estimated lift (%), SCM", f"{est_pct*100:.2f}%")
            c3.metric("Error (pp)", f"{(est_pct-true_pct)*100:.2f} pp")




st.subheader("Method results")
res_show = results.filter(pl.col("campaign_id") == campaign_id).sort("method")
st.dataframe(res_show.to_pandas(), use_container_width=True)

es_path = processed_dir / f"event_study_{campaign_id}.parquet"
st.subheader("Event study (optional)")
if es_path.exists():
    es = pl.read_parquet(es_path).sort("rel_day")
    st.dataframe(es.to_pandas(), use_container_width=True)
else:
    st.caption(f"No event study file found at {es_path}.")
