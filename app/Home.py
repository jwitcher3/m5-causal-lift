from __future__ import annotations
import io
from datetime import date 
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

# --- Campaign overview (derived from ground truth) ---
gt_path = processed_dir / "fact_ground_truth.parquet"
camp_meta = None
if gt_path.exists():
    gt_c = (
        pl.read_parquet(gt_path)
        .filter(pl.col("campaign_id") == campaign_id)
        .select(["date", "treated", "in_campaign"])
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("treated").cast(pl.Int8),
            pl.col("in_campaign").cast(pl.Int8),
        )
    )

    # campaign window from in_campaign dates
    win = gt_c.filter(pl.col("in_campaign") == 1).select(
        pl.col("date").min().alias("start"),
        pl.col("date").max().alias("end"),
    )
    start_dt = win["start"][0]
    end_dt = win["end"][0]

    # treated/control unit counts (at selected grain)
    # infer grain by checking columns in results file naming or just default store_dept
    # if you want exact: treat unit key is whatever simulator used; for now count by rows grouped in GT is heavy.
    # simple approximation: count treated rows / unique treated days is not ideal, so keep it light:
    treated_rows = int(gt_c.filter(pl.col("treated") == 1).height)
    treated_in_rows = int(gt_c.filter((pl.col("treated") == 1) & (pl.col("in_campaign") == 1)).height)

    camp_meta = {
        "start": start_dt,
        "end": end_dt,
        "treated_rows": treated_rows,
        "treated_in_campaign_rows": treated_in_rows,
    }

st.subheader("Campaign overview")
if camp_meta is None or camp_meta["start"] is None:
    st.caption("No ground truth found for this campaign (run simulator first).")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Campaign window", f"{camp_meta['start']} → {camp_meta['end']}")
    c2.metric("Treated rows", f"{camp_meta['treated_rows']:,}")
    c3.metric("Treated rows (in-campaign)", f"{camp_meta['treated_in_campaign_rows']:,}")

st.sidebar.subheader("Run commands")
if camp_meta and camp_meta["start"] and camp_meta["end"]:
    cmds = f"""# End-to-end for this campaign
make simulate CAMPAIGN_ID={campaign_id} START_DATE={camp_meta['start']} END_DATE={camp_meta['end']} TREAT_FRAC=0.2 MAX_UPLIFT=0.15 SEED=7
make did CAMPAIGN_ID={campaign_id}
make scm CAMPAIGN_ID={campaign_id} USE_LOG1P=1 DONOR_GRAIN=store_dept ALPHA=50
make eval
make app
"""
else:
    cmds = """# End-to-end
make pipeline
make app
"""
st.sidebar.code(cmds, language="bash")


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



# --- Choose best DiD method automatically (unit ATT) ---
best_did_row = None
if ev is not None:
    best_did = (
        ev.filter(
            (pl.col("method") == "twfe_did")  # tighten if you add more DiD variants later
            & (pl.col("truth_used") == "unit_att")
        )
        .sort("abs_bias")
    )
    if best_did.height > 0:
        best_did_row = best_did.row(0, named=True)

st.subheader("Overview")

c1, c2, c3, c4 = st.columns(4)

# SCM KPI (percent lift)
if best_scm_row is not None:
    true_pct = float(best_scm_row.get("att_true_pct"))
    est_pct = None
    if best_scm_row.get("att_hat_pct") is not None:
        est_pct = float(best_scm_row["att_hat_pct"])
    elif best_scm_row.get("truth_used") == "log1p_att":
        est_pct = float(np.expm1(float(best_scm_row["att_hat_used"])))

    c1.metric("True lift (%)", f"{true_pct*100:.2f}%")
    if est_pct is not None:
        c2.metric("SCM est lift (%)", f"{est_pct*100:.2f}%")
        c3.metric("SCM error (pp)", f"{(est_pct-true_pct)*100:.2f} pp")
    c4.metric("SCM method", str(best_scm_row.get("method")))
else:
    c1.metric("True lift (%)", "—")
    c2.metric("SCM est lift (%)", "—")
    c3.metric("SCM error (pp)", "—")
    c4.metric("SCM method", "—")

# DiD KPI (unit ATT)
if best_did_row is not None:
    st.caption(f"DiD (unit ATT): est={best_did_row['att_hat_used']:.2f} vs true={best_did_row['att_true_used']:.2f} (bias={best_did_row['bias']:.2f})")


if ev is not None:
    best = (
    ev.filter(
        (pl.col("method").str.contains("^scm_"))
        & (pl.col("truth_used") == "log1p_att")
        & (~pl.col("method").str.contains("simplex"))
    )
    .filter(~pl.col("method").str.contains("log1p_sim"))  # <-- add this
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

st.subheader("Downloads")
c1, c2 = st.columns(2)

if ev is not None and ev.height > 0:
    buf = io.StringIO()
    ev.write_csv(buf)
    c1.download_button(
        "Download eval (CSV)",
        data=buf.getvalue().encode("utf-8"),
        file_name=f"eval_{campaign_id}.csv",
        mime="text/csv",
    )

res_show = results.filter(pl.col("campaign_id") == campaign_id).sort("method")
buf2 = io.StringIO()
res_show.write_csv(buf2)
c2.download_button(
    "Download method results (CSV)",
    data=buf2.getvalue().encode("utf-8"),
    file_name=f"method_results_{campaign_id}.csv",
    mime="text/csv",
)
