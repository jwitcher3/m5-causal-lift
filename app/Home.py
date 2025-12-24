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

st.sidebar.subheader("Experiment runner")

new_cid = st.sidebar.text_input("new_campaign_id", "cmp_003")
start_str = st.sidebar.text_input("start_date (YYYY-MM-DD)", "2014-08-01")
end_str = st.sidebar.text_input("end_date (YYYY-MM-DD)", "2014-08-28")
treat_frac = st.sidebar.slider("treat_frac", 0.05, 0.50, 0.20, 0.05)
max_uplift = st.sidebar.slider("max_uplift", 0.01, 0.30, 0.15, 0.01)
seed = st.sidebar.number_input("seed", min_value=1, max_value=10_000, value=7)

cmds = f"""# Run a fresh campaign end-to-end
make simulate CAMPAIGN_ID={new_cid} START_DATE={start_str} END_DATE={end_str} TREAT_FRAC={treat_frac} MAX_UPLIFT={max_uplift} SEED={seed}
make did CAMPAIGN_ID={new_cid}
make scm CAMPAIGN_ID={new_cid} USE_LOG1P=1 DONOR_GRAIN=store_dept ALPHA=50
make eval
"""
st.sidebar.code(cmds, language="bash")


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
    #ev_show = ev.filter(~pl.col("method").str.contains("simplex")).sort("abs_bias")
    ev_show = (
    ev.filter(~pl.col("method").str.contains("simplex"))
      .filter(~pl.col("method").str.contains("log1p_sim"))
      .sort("abs_bias")
    )   

    st.dataframe(ev_show.to_pandas(), use_container_width=True)
else:
    st.caption("No eval table found. Run: python src/m5lift/eval/evaluate.py")

st.subheader("Error summary (abs bias)")
ev_chart = (
    ev_show.select(["method", "abs_bias"])
           .to_pandas()
           .set_index("method")
)
st.bar_chart(ev_chart, use_container_width=True)

st.subheader("Leaderboard (all campaigns)")

all_eval_path = processed_dir / "fact_method_eval.parquet"
if all_eval_path.exists():
    all_ev = pl.read_parquet(all_eval_path).with_columns(pl.col("bias").abs().alias("abs_bias"))

    # optional filters (keeps it clean)
    hide_simplex = st.checkbox("Hide simplex", value=True)
    hide_sim = st.checkbox("Hide log1p_sim rows", value=True)

    if hide_simplex:
        all_ev = all_ev.filter(~pl.col("method").str.contains("simplex"))
    if hide_sim:
        all_ev = all_ev.filter(~pl.col("method").str.contains("log1p_sim"))

    leaderboard = (
        all_ev.group_by("method")
        .agg(
            pl.len().alias("n_campaigns"),
            pl.mean("abs_bias").alias("mean_abs_bias"),
            pl.median("abs_bias").alias("median_abs_bias"),
            pl.mean("rel_bias").alias("mean_rel_bias"),
        )
        .sort("mean_abs_bias")
    )

    st.dataframe(leaderboard.to_pandas(), use_container_width=True)

    lb_chart = leaderboard.select(["method", "mean_abs_bias"]).to_pandas().set_index("method")
    st.bar_chart(lb_chart, use_container_width=True)
else:
    st.caption("No fact_method_eval.parquet yet. Run: make eval")

st.subheader("Diagnostics (where available)")

diag = pl.read_parquet(all_eval_path)

# keep only rows with diagnostics
diag = diag.filter(
    (pl.col("pretrend_p").is_not_null()) | (pl.col("rmse_pre").is_not_null())
).with_columns(pl.col("bias").abs().alias("abs_bias"))

# show table first
st.dataframe(
    diag.select(["campaign_id","method","abs_bias","pretrend_p","rmse_pre"]).to_pandas(),
    use_container_width=True
)

# simple charts (Streamlit native)
if diag.select(pl.col("pretrend_p").is_not_null().any()).item():
    pchart = diag.filter(pl.col("pretrend_p").is_not_null()).select(["method","pretrend_p"]).to_pandas().set_index("method")
    st.caption("Pretrend p-values by method (lower can indicate violation risk)")
    st.bar_chart(pchart, use_container_width=True)

if diag.select(pl.col("rmse_pre").is_not_null().any()).item():
    rchart = diag.filter(pl.col("rmse_pre").is_not_null()).select(["method","rmse_pre"]).to_pandas().set_index("method")
    st.caption("Pre-period fit RMSE (lower is better on the fit scale)")
    st.bar_chart(rchart, use_container_width=True)

st.sidebar.subheader("Campaign sweep")
n = st.sidebar.number_input("n_campaigns", 1, 20, 5)
base_seed = st.sidebar.number_input("base_seed", 1, 10000, 100)

sweep = "\n".join([
    f"make simulate CAMPAIGN_ID=cmp_{i:03d} START_DATE={start_str} END_DATE={end_str} "
    f"TREAT_FRAC={treat_frac} MAX_UPLIFT={max_uplift} SEED={base_seed+i}\n"
    f"make did CAMPAIGN_ID=cmp_{i:03d}\n"
    f"make scm CAMPAIGN_ID=cmp_{i:03d} USE_LOG1P=1 DONOR_GRAIN=store_dept ALPHA=50\n"
    for i in range(1, n+1)
]) + "\nmake eval"

st.sidebar.code(sweep, language="bash")


# --- Pick best SCM + best DiD from eval (so Overview has real values) ---
best_scm_method = None
best_scm_row = None
best_did_row = None

if ev is not None and ev.height > 0:
    # Best SCM (prefer log1p truth; exclude simplex + the known-bad sim row)
    best_scm = (
        ev.filter(
            (pl.col("method").str.contains("^scm_"))
            & (pl.col("truth_used") == "log1p_att")
            & (~pl.col("method").str.contains("simplex"))
            & (~pl.col("method").str.contains("log1p_sim"))
        )
        .sort("abs_bias")
    )
    if best_scm.height > 0:
        best_scm_row = best_scm.row(0, named=True)
        best_scm_method = best_scm_row["method"]

    # Best DiD (unit ATT)
    best_did = (
        ev.filter(
            (pl.col("method") == "twfe_did")
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
    true_pct = float(best_scm_row["att_true_pct"])
    if best_scm_row.get("att_hat_pct") is not None:
        est_pct = float(best_scm_row["att_hat_pct"])
    else:
        # If evaluated on log1p scale, convert to pct
        est_pct = float(np.expm1(float(best_scm_row["att_hat_used"])))

    c1.metric("True lift (%)", f"{true_pct*100:.2f}%")
    c2.metric("SCM est lift (%)", f"{est_pct*100:.2f}%")
    c3.metric("SCM error (pp)", f"{(est_pct-true_pct)*100:.2f} pp")
    c4.metric("SCM method", str(best_scm_row["method"]))
else:
    c1.metric("True lift (%)", "—")
    c2.metric("SCM est lift (%)", "—")
    c3.metric("SCM error (pp)", "—")
    c4.metric("SCM method", "—")

# DiD KPI (unit ATT)
if best_did_row is not None:
    st.caption(
        f"DiD (unit ATT): est={best_did_row['att_hat_used']:.2f} "
        f"vs true={best_did_row['att_true_used']:.2f} "
        f"(bias={best_did_row['bias']:.2f}, pretrend_p={best_did_row.get('pretrend_p')})"
    )



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
st.subheader("Event study")
if es_path.exists():
    es = pl.read_parquet(es_path).sort("rel_day")
    es_pd = es.to_pandas().set_index("rel_day")

    if "beta" in es_pd.columns:
        st.line_chart(es_pd[["beta"]], use_container_width=True)
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
