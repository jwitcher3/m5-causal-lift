from __future__ import annotations
import io
from datetime import date 
from pathlib import Path
import polars as pl
import streamlit as st
import numpy as np

def safe_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def grade_from_flags(n_flags: int) -> str:
    if n_flags <= 0:
        return "A"
    if n_flags == 1:
        return "B"
    return "C"

def recommendation_from_grade(g: str) -> str:
    return {"A": "Ship", "B": "Monitor", "C": "Don’t ship"}.get(g, "Monitor")

def compute_scm_stability(ts: pl.DataFrame, lift_col: str = "lift_hat_units") -> dict:
    inc = ts.filter(pl.col("in_campaign") == 1)
    if inc.height == 0 or lift_col not in inc.columns:
        return {"lift_mean": None, "lift_std": None, "cv": None}

    lift_mean = safe_float(inc.select(pl.col(lift_col).mean()).item())
    lift_std  = safe_float(inc.select(pl.col(lift_col).std()).item())
    denom = abs(lift_mean) if lift_mean is not None and abs(lift_mean) > 1e-9 else None
    cv = (lift_std / denom) if (lift_std is not None and denom is not None) else None
    return {"lift_mean": lift_mean, "lift_std": lift_std, "cv": cv}


def grade_scm(rmse_pre: float | None, cv: float | None) -> tuple[str, list[str]]:
    """
    Simple retail-friendly guardrails.
    - rmse_pre is on fit scale (log1p) if you ran log1p SCM, otherwise units.
    """
    flags = []
    # thresholds you can tune after you see a few campaigns
    if rmse_pre is None:
        flags.append("Missing pre-fit metric (rmse_pre).")
    else:
        # log1p rmse is typically ~0.05–0.35 in your runs
        if rmse_pre > 0.35:
            flags.append(f"Weak pre-fit (rmse_pre={rmse_pre:.3f}).")

    if cv is None:
        flags.append("Missing stability metric (CV).")
    else:
        if cv > 2.0:
            flags.append(f"Unstable daily lift (CV={cv:.2f}).")

    g = grade_from_flags(len(flags))
    return g, flags

def grade_did(pretrend_p: float | None) -> tuple[str, list[str]]:
    flags = []
    if pretrend_p is None:
        flags.append("Missing pre-trend test (pretrend_p).")
    else:
        if pretrend_p < 0.05:
            flags.append(f"Pre-trend violation risk (p={pretrend_p:.3f}).")
        elif pretrend_p < 0.10:
            flags.append(f"Borderline pre-trend (p={pretrend_p:.3f}).")

    g = grade_from_flags(len(flags))
    return g, flags


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
ev_show = None

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

    st.dataframe(ev_show.to_pandas(), width="stretch")
else:
    st.caption("No eval table found. Run: python src/m5lift/eval/evaluate.py")

st.subheader("Error summary (abs bias)")
if ev_show is None or ev_show.height == 0:
    st.caption("No eval rows to chart yet. Run: make eval")
else:
    ev_chart = ev_show.select(["method", "abs_bias"]).to_pandas().set_index("method")
    st.bar_chart(ev_chart, width="stretch")


st.subheader("Scorecard (across campaigns)")

all_eval_path = processed_dir / "fact_method_eval.parquet"
if not all_eval_path.exists():
    st.caption("No eval table yet. Run: make eval")
else:
    all_ev = pl.read_parquet(all_eval_path).with_columns(
        pl.col("bias").abs().alias("abs_bias"),
        pl.col("rel_bias").abs().alias("abs_rel_bias"),
    )

    c1, c2, c3 = st.columns(3)

    hide_simplex = st.checkbox("Hide simplex", value=True, key="sc_hide_simplex")
    hide_sim = st.checkbox("Hide log1p_sim", value=True, key="sc_hide_log1p_sim")
    min_campaigns = st.number_input("Min campaigns per method", min_value=1, max_value=100, value=1, key="sc_min_campaigns")


    if hide_simplex:
        all_ev = all_ev.filter(~pl.col("method").str.contains("simplex"))
    if hide_sim:
        all_ev = all_ev.filter(~pl.col("method").str.contains("log1p_sim"))

    # Winner per campaign (lowest abs_bias)
    winners = (
        all_ev.sort(["campaign_id", "abs_bias"])
        .group_by("campaign_id")
        .agg(pl.col("method").first().alias("winner_method"))
    )

    all_ev = all_ev.join(winners, on="campaign_id", how="left").with_columns(
        (pl.col("method") == pl.col("winner_method")).cast(pl.Int8).alias("is_winner")
    )

    scorecard = (
        all_ev.group_by("method")
        .agg(
            pl.len().alias("n_rows"),
            pl.n_unique("campaign_id").alias("n_campaigns"),
            pl.mean("abs_bias").alias("mean_abs_bias"),
            pl.median("abs_bias").alias("median_abs_bias"),
            pl.mean("abs_rel_bias").alias("mean_abs_rel_bias"),
            pl.mean("is_winner").alias("win_rate"),
        )
        .filter(pl.col("n_campaigns") >= min_campaigns)
        .sort(["mean_abs_bias", "median_abs_bias"])
    )

    st.dataframe(scorecard.to_pandas(), width="stretch")

    # Quick chart: mean_abs_bias by method
    chart_df = scorecard.select(["method", "mean_abs_bias"]).to_pandas().set_index("method")
    st.bar_chart(chart_df, width="stretch")

    # Optional: winners table
    with st.expander("Winners by campaign"):
        st.dataframe(
            winners.sort("campaign_id").to_pandas(),
            width="stretch"
        )


st.subheader("Diagnostics (where available)")
if not all_eval_path.exists():
    st.caption("No eval table yet. Run: make eval")
else:
    diag = pl.read_parquet(all_eval_path)
    diag = diag.filter(
        (pl.col("pretrend_p").is_not_null()) | (pl.col("rmse_pre").is_not_null())
    ).with_columns(pl.col("bias").abs().alias("abs_bias"))

    st.dataframe(
        diag.select(["campaign_id","method","abs_bias","pretrend_p","rmse_pre"]).to_pandas(),
        width="stretch"
    )

# simple charts (Streamlit native)
    if diag.select(pl.col("pretrend_p").is_not_null().any()).item():
        pchart = diag.filter(pl.col("pretrend_p").is_not_null()).select(["method","pretrend_p"]).to_pandas().set_index("method")
        st.caption("Pretrend p-values by method (lower can indicate violation risk)")
        st.bar_chart(pchart, width="stretch")

    if diag.select(pl.col("rmse_pre").is_not_null().any()).item():
        rchart = diag.filter(pl.col("rmse_pre").is_not_null()).select(["method","rmse_pre"]).to_pandas().set_index("method")
        st.caption("Pre-period fit RMSE (lower is better on the fit scale)")
        st.bar_chart(rchart, width="stretch")

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
ts = None
lift_col = "lift_hat"  # safe default

if series_file:
    series_path = processed_dir / series_file
    ts = pl.read_parquet(series_path).sort("date")
    ts_pd = ts.to_pandas().set_index("date")

    # choose which lift column exists in the parquet
    lift_col = "lift_hat_units" if "lift_hat_units" in ts.columns else "lift_hat"

    st.subheader("Synthetic control: treated vs counterfactual")
    st.line_chart(ts_pd[["y_treated", "y0_hat"]], width="stretch")

    st.subheader("Synthetic control: estimated lift over time")
    st.line_chart(ts_pd[[lift_col]], width="stretch")

    # Metrics (true vs estimated) from eval row if available
    if best_scm_row is not None:
        true_pct = float(best_scm_row.get("att_true_pct"))
        est_pct = None

        if best_scm_row.get("att_hat_pct") is not None:
            est_pct = float(best_scm_row["att_hat_pct"])
        else:
            if best_scm_row.get("truth_used") == "log1p_att":
                est_pct = float(np.expm1(float(best_scm_row["att_hat_used"])))

        if est_pct is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("True lift (%), log1p truth", f"{true_pct*100:.2f}%")
            c2.metric("Estimated lift (%), SCM", f"{est_pct*100:.2f}%")
            c3.metric("Error (pp)", f"{(est_pct-true_pct)*100:.2f} pp")

    # --- Donor weights (if available) ---
    # infer weights filename from the selected series_file name
    # e.g. scm_series_cmp_001_store_dept_log1p.parquet -> scm_weights_cmp_001_store_dept_log1p.parquet
    weights_file = series_file.replace("scm_series_", "scm_weights_")
    weights_path = processed_dir / weights_file

    with st.expander("SCM donor weights (interpretability)", expanded=False):
        if weights_path.exists():
            wdf = pl.read_parquet(weights_path)

            # helpful derived fields
            wdf = wdf.with_columns(
                pl.col("weight").abs().alias("abs_weight")
            ).sort("abs_weight", descending=True)

            top_n = st.slider("Show top N donors", 5, 50, 15, key="scm_top_n_donors")

            st.dataframe(
                wdf.select(["donor_id", "weight", "abs_weight"]).head(top_n).to_pandas(),
                width="stretch"
            )

            # quick chart
            wchart = (
                wdf.select(["donor_id", "abs_weight"])
                   .head(top_n)
                   .to_pandas()
                   .set_index("donor_id")
            )
            st.bar_chart(wchart, width="stretch")
        else:
            st.caption(f"No weights file found at {weights_path.name}. Re-run: make scm CAMPAIGN_ID={campaign_id} ...")


st.subheader("Decision (units)")

scm_rmse = safe_float(best_scm_row.get("rmse_pre")) if best_scm_row else None
did_p = safe_float(best_did_row.get("pretrend_p")) if best_did_row else None

# Default stability unknown until we have ts
scm_cv = None
scm_stability = None

if ts is not None:
    scm_stability = compute_scm_stability(ts, lift_col=lift_col)
    scm_cv = scm_stability["cv"]

# Grade once (after stability computed if available)
scm_grade, scm_flags = grade_scm(scm_rmse, scm_cv)
did_grade, did_flags = grade_did(did_p)

c1, c2, c3 = st.columns(3)
c1.metric("SCM confidence", f"{scm_grade} — {recommendation_from_grade(scm_grade)}")
c2.metric("DiD confidence", f"{did_grade} — {recommendation_from_grade(did_grade)}")
c3.metric("Primary KPI", "Units")

if scm_flags:
    st.warning("SCM checks:\n- " + "\n- ".join(scm_flags))
if did_flags:
    st.warning("DiD checks:\n- " + "\n- ".join(did_flags))

if scm_stability and scm_stability["cv"] is not None:
    st.caption(
        f"SCM stability (in-campaign): mean lift={scm_stability['lift_mean']:.2f} units, "
        f"std={scm_stability['lift_std']:.2f}, CV={scm_stability['cv']:.2f}"
    )
else:
    st.caption("SCM stability: not available (missing series or no in-campaign days).")


st.subheader("Method results")
res_show = results.filter(pl.col("campaign_id") == campaign_id).sort("method")
st.dataframe(res_show.to_pandas(), width="stretch")

es_path = processed_dir / f"event_study_{campaign_id}.parquet"
st.subheader("Event study")
if es_path.exists():
    es = pl.read_parquet(es_path).sort("rel_day")
    es_pd = es.to_pandas().set_index("rel_day")

    if "beta" in es_pd.columns:
        st.line_chart(es_pd[["beta"]], width="stretch")
    st.dataframe(es.to_pandas(), width="stretch")
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

st.subheader("Method health metrics (rmse + pretrend)")

if ev is None:
    st.caption("Run eval first (make eval).")
else:
    # Use the eval table across all campaigns (load full eval file)
    ev_all = pl.read_parquet(eval_path)

    # Filter out method variants you don’t want to rank
    ev_all = (
        ev_all
        .filter(~pl.col("method").str.contains("simplex"))
        .filter(~pl.col("method").str.contains("log1p_sim"))
    )

    # Build a method-level leaderboard:
    # - SCM: avg rmse_pre (lower better)
    # - DiD: share with pretrend_p >= 0.10 (higher better)
    # - Coverage: number of campaigns where method exists
    leaderboard = (
        ev_all.group_by("method")
        .agg(
            pl.len().alias("n_campaigns"),
            pl.col("rmse_pre").mean().alias("rmse_pre_avg"),
            (pl.col("pretrend_p") >= 0.10).mean().alias("pretrend_pass_rate"),
        )
        .with_columns(
            pl.col("rmse_pre_avg").fill_null(999.0),
            pl.col("pretrend_pass_rate").fill_null(0.0),
        )
        .sort(["n_campaigns", "rmse_pre_avg"], descending=[True, False])
    )

    st.dataframe(leaderboard.to_pandas(), width="stretch")


res_show = results.filter(pl.col("campaign_id") == campaign_id).sort("method")
buf2 = io.StringIO()
res_show.write_csv(buf2)
c2.download_button(
    "Download method results (CSV)",
    data=buf2.getvalue().encode("utf-8"),
    file_name=f"method_results_{campaign_id}.csv",
    mime="text/csv",
)
