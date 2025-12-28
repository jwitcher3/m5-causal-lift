from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placebo Tests", layout="wide")
st.title("Placebo Tests (SCM)")
st.caption("Run SCM on many fake treatment windows to see how ‘rare’ your observed lift is.")

DEFAULT_PROCESSED = Path("data/processed")
processed_dir = Path(st.sidebar.text_input("processed_dir", str(DEFAULT_PROCESSED)))

results_path = processed_dir / "fact_method_results.parquet"
if not results_path.exists():
    st.error(f"Missing {results_path}. Run simulator + methods first.")
    st.stop()

results = pl.read_parquet(results_path)
campaigns = sorted(results["campaign_id"].unique().to_list())
campaign_id = st.sidebar.selectbox("campaign_id", campaigns, index=0)

donor_grain = st.sidebar.selectbox("donor_grain", ["store", "store_dept"], index=1)
use_log1p = st.sidebar.checkbox("use_log1p", value=True)

suffix = "_log1p" if use_log1p else ""
placebo_file = processed_dir / f"scm_placebo_{campaign_id}_{donor_grain}{suffix}.parquet"

st.subheader("Inputs")
st.write(
    {
        "campaign_id": campaign_id,
        "donor_grain": donor_grain,
        "use_log1p": use_log1p,
        "placebo_file": placebo_file.name,
    }
)

if not placebo_file.exists():
    st.warning("No placebo file found for this selection.")
    st.code(
        f"""python src/m5lift/methods/placebo_scm.py \\
  --processed_dir {processed_dir} \\
  --campaign_id {campaign_id} \\
  --donor_grain {donor_grain} \\
  {"--use_log1p \\\n  " if use_log1p else ""}--alpha 50 \\
  --n_placebos 50"""
    )
    st.stop()

plc = pl.read_parquet(placebo_file)

if "att_hat_units" not in plc.columns:
    st.error(f"Expected att_hat_units in placebo file, got columns: {plc.columns}")
    st.stop()

# Try to find the “real” SCM estimate from fact_method_results
method_prefix = f"scm_ridge_{donor_grain}"
method = method_prefix + ("_log1p" if use_log1p else "")

real_att_units = None
real_row = (
    results.filter((pl.col("campaign_id") == campaign_id) & (pl.col("method") == method))
    .select(["att_hat", "method"])
)
if real_row.height > 0 and "att_hat" in real_row.columns:
    real_att_units = float(real_row["att_hat"][0])

st.subheader("Summary")
abs_plc = plc.select(pl.col("att_hat_units").abs().alias("abs_att")).to_series().to_numpy()
plc_att = plc["att_hat_units"].to_numpy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("N placebos", f"{plc.height:,}")
c2.metric("Mean placebo (units)", f"{float(np.mean(plc_att)):.2f}")
c3.metric("Std placebo (units)", f"{float(np.std(plc_att, ddof=1)):.2f}")
c4.metric("Median placebo (units)", f"{float(np.median(plc_att)):.2f}")

if real_att_units is not None:
    p_two_sided = float(np.mean(np.abs(plc_att) >= abs(real_att_units)))
    st.success(f"Observed SCM ATT (units): **{real_att_units:.2f}** • placebo p-value (two-sided): **{p_two_sided:.3f}**")
else:
    st.warning(
        f"Could not find real SCM row in fact_method_results for method='{method}'. "
        "Placebo distribution is still shown."
    )

# Histogram (binned) using st.bar_chart (no width arg)
st.subheader("Placebo distribution (ATT in units)")
bins = st.slider("Histogram bins", 10, 80, 30)
counts, edges = np.histogram(plc_att, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2.0
hist_df = pd.DataFrame({"att_bin_center": centers, "count": counts}).set_index("att_bin_center")
st.bar_chart(hist_df)

# Quantiles and outliers
st.subheader("Placebo quantiles")
q = plc.select(
    pl.len().alias("n"),
    pl.col("att_hat_units").quantile(0.05).alias("p05"),
    pl.col("att_hat_units").quantile(0.25).alias("p25"),
    pl.col("att_hat_units").median().alias("p50"),
    pl.col("att_hat_units").quantile(0.75).alias("p75"),
    pl.col("att_hat_units").quantile(0.95).alias("p95"),
)
st.dataframe(q.to_pandas(), use_container_width=True)

st.subheader("Largest absolute placebo lifts")
top_n = st.slider("Show top N absolute placebo lifts", 5, 50, 15)
top = (
    plc.with_columns(pl.col("att_hat_units").abs().alias("abs_att_hat_units"))
    .sort("abs_att_hat_units", descending=True)
    .head(top_n)
)
st.dataframe(
    top.select(
        [
            "placebo_start",
            "placebo_end",
            "att_hat_units",
            "abs_att_hat_units",
            "rmse_pre",
            "cv",
            "n_pre_days",
            "n_placebo_days",
        ]
    ).to_pandas(),
    use_container_width=True,
)

st.subheader("Download")
csv_bytes = plc.write_csv().encode("utf-8")
st.download_button(
    "Download placebo results (CSV)",
    data=csv_bytes,
    file_name=f"scm_placebo_{campaign_id}_{donor_grain}{suffix}.csv",
    mime="text/csv",
)
