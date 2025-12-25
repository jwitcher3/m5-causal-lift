import streamlit as st

st.set_page_config(page_title="Documentation", layout="wide")

st.title("Documentation")
st.caption("What this project does, how to run it, and what artifacts it produces.")

st.markdown(
    """
## What this app is
This is an **incrementality sandbox** built on **M5-style simulated retail data**.
It simulates a campaign (treatment vs control) and evaluates causal methods against known ground truth.

## How to run (typical flow)
1. **Simulate** a campaign:
   - `make simulate CAMPAIGN_ID=cmp_003 START_DATE=2014-08-01 END_DATE=2014-08-28 TREAT_FRAC=0.2 MAX_UPLIFT=0.15 SEED=7`
2. **Run DiD / Event Study**:
   - `make did CAMPAIGN_ID=cmp_003`
3. **Run Synthetic Control (SCM)**:
   - `make scm CAMPAIGN_ID=cmp_003 USE_LOG1P=1 DONOR_GRAIN=store_dept ALPHA=50`
4. **Evaluate vs ground truth**:
   - `make eval`
5. **Launch app**:
   - `make app` (or `streamlit run app/Home.py`)

## Files produced (data/processed)
- `fact_ground_truth.parquet`  
  Ground truth panel including treatment assignment, true effect (`tau`), outcomes (`y_obs`), etc.

- `fact_method_results.parquet`  
  One row per method per campaign (ATT estimates, bias, diagnostics such as `rmse_pre`, etc.).

- `fact_method_eval.parquet`  
  “Scale-aware” evaluation table used by the dashboard for ranking methods.

- `scm_series_<campaign>_<grain>[_log1p].parquet`  
  Time series for SCM plots: `y_treated`, `y0_hat`, and lift columns (units + alias).

- `scm_weights_<campaign>_<grain>[_log1p].parquet`  
  Donor weights for interpretability (used by the Home dashboard expander).

## Synthetic Control (SCM) notes
- SCM is implemented as ridge regression on donor time series in the **pre-period**.
- Optionally runs on **log1p scale** for stability, then converts back to units for ATT.
- The dashboard reports:
  - Fit RMSE on fit scale (`rmse_pre`)
  - In-campaign lift stability (CV) on units scale
  - Donor weights for interpretability

## Troubleshooting
- “Missing fact_method_results.parquet” → run `make simulate` + `make did`/`make scm` first.
- No SCM series files → run `make scm ...` for the selected campaign.
- No weights file → rerun SCM after this feature was added.

---
Tip: Keep this page lightweight. The README can hold deeper documentation later.
"""
)
