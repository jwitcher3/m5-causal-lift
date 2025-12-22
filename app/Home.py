from __future__ import annotations

from pathlib import Path
import polars as pl
import streamlit as st

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

st.subheader("Method results")
res_show = results.filter(pl.col("campaign_id") == campaign_id).sort("method")
st.dataframe(res_show.to_pandas(), use_container_width=True)

st.sidebar.subheader("SCM series")
series_options = [
    f"scm_series_{campaign_id}_store.parquet",
    f"scm_series_{campaign_id}_store_dept.parquet",
    f"scm_series_{campaign_id}_store_dept_log1p.parquet",
]
series_file = st.sidebar.selectbox("series file", series_options, index=min(2, len(series_options)-1))
series_path = processed_dir / series_file

if series_path.exists():
    ts = pl.read_parquet(series_path).sort("date")
    ts_pd = ts.to_pandas().set_index("date")

    st.subheader("Treated vs counterfactual")
    st.line_chart(ts_pd[["y_treated", "y0_hat"]], use_container_width=True)

    st.subheader("Estimated lift over time")
    st.line_chart(ts_pd[["lift_hat"]], use_container_width=True)
else:
    st.warning(f"Missing {series_path}. Run synth_control.py to generate this series file.")

es_path = processed_dir / f"event_study_{campaign_id}.parquet"
st.subheader("Event study (optional)")
if es_path.exists():
    es = pl.read_parquet(es_path).sort("rel_day")
    st.dataframe(es.to_pandas(), use_container_width=True)
else:
    st.caption(f"No event study file found at {es_path}.")
