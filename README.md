# M5 Causal Lift (Incrementality Sandbox)

A small, end-to-end **incrementality sandbox** using **M5-style simulated retail data**.  
It simulates treatment/control campaigns and evaluates causal methods against **known ground truth**.

This repo is intentionally “product-shaped”:
- a reproducible pipeline (`make demo`, `make pipeline`)
- method outputs saved as Parquet artifacts
- a Streamlit app to explore results + diagnostics
- a trust checklist (guardrails + placebo tests)

---

## What you get
- **Campaign simulator** (creates treated/control + true effect)
- **Methods**
  - DiD / event study
  - Synthetic Control (ridge SCM, optional log1p fit)
- **Evaluation vs ground truth** (scale-aware)
- **Streamlit app** to explore results, scorecard, diagnostics, and SCM interpretability (donor weights)
- **Method trust checks**
  - Pre-fit RMSE + stability (CV)
  - **Placebo tests** (fake treatment dates) + placebo lift distribution

---

## Repo layout

```text
app/
  Home.py
  pages/
    01_Documentation.py
    02_method_trust_checklist.py
    03_Placebo Tests.py
data/
  raw/
  processed/              # generated artifacts (parquet)
scripts/
  demo.sh
  smoke.sh
src/
  m5lift/
    sim/
    methods/
    eval/
    io/
````

---

## Quickstart

### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Demo (one command)

Runs the full pipeline + launches the app.

```bash
./scripts/demo.sh
```

Then open:

* [http://localhost:8501](http://localhost:8501)

### 3) Demo (Makefile version)

```bash
make install
make demo
make app
```

---

## Core commands

### End-to-end pipeline

```bash
make pipeline \
  CAMPAIGN_ID=cmp_003 \
  START_DATE=2014-08-01 \
  END_DATE=2014-08-28 \
  TREAT_FRAC=0.2 \
  MAX_UPLIFT=0.15 \
  SEED=7 \
  DONOR_GRAIN=store_dept \
  USE_LOG1P=1 \
  ALPHA=50
```

### Placebo tests (fake treatment windows)

```bash
make placebo \
  CAMPAIGN_ID=cmp_003 \
  DONOR_GRAIN=store_dept \
  USE_LOG1P=1 \
  ALPHA=50 \
  N_PLACEBOS=50 \
  MIN_PRE_DAYS=28 \
  SEED=7
```

### Run the app

```bash
make app
```

### Verify artifacts exist (minimal hardening)

```bash
make check
```

### Smoke test

```bash
./scripts/smoke.sh
```

---

## Artifacts written to `data/processed/`

### Ground truth

* `fact_ground_truth.parquet`
  Ground truth panel including treatment assignment, true effect (`tau`), and observed outcomes (`y_obs`).

### Method outputs

* `fact_method_results.parquet`
  One row per `(campaign_id, method)` with:

  * `att_hat` (estimated ATT in units)
  * SCM diagnostics like `rmse_pre`, plus other method diagnostics where available

* `fact_method_eval.parquet`
  Scale-aware evaluation table used by the dashboard (e.g., bias vs true ATT).

### SCM series + interpretability

* `scm_series_<campaign>_<grain>[_log1p].parquet`
  Time series used for SCM plots:

  * `y_treated`, `y0_hat`, and lift columns (commonly `lift_hat_units`)

* `scm_weights_<campaign>_<grain>[_log1p].parquet`
  Donor weights (interpretability / “what built the counterfactual”).

### Placebo results

* `scm_placebo_<campaign>_<grain>[_log1p].parquet`
  Placebo windows + outcomes:

  * `placebo_start`, `placebo_end`
  * `att_hat_units`
  * `rmse_pre`, `cv`

---

## How to interpret results

### Key definitions

* **ATT (units):** average incremental lift in the campaign window
  (positive = incremental demand, negative = possible cannibalization or model mismatch)
* **Counterfactual (`y0_hat`):** what SCM estimates would have happened without treatment
* **Lift series (`lift_hat_units`):** daily `y_treated - y0_hat`

### DiD / Event Study

* **Assumption:** parallel trends between treated and control
* **Diagnostic:** `pretrend_p`
  Smaller p-values can signal pre-trend differences (violation risk)

### Synthetic Control (SCM)

* **Pre-fit RMSE (`rmse_pre`):** how well donors match treated pre-period (lower is better)
* **Stability CV (`cv`):** variability of daily lift within the campaign (lower is more stable)
* **Donor weights:** sanity check that counterfactual isn’t dominated by strange donors

---

## Placebo tests (fake treatment dates)

**Why:** detect timing artifacts where SCM would “find lift” even when nothing happened.

**How it works:**

* sample placebo windows before the real campaign window
* run SCM as if treatment started then
* compare real ATT vs placebo ATT distribution

**Empirical two-sided p-value:**

```text
p = mean(|ATT_placebo| >= |ATT_real|)
```

**Rule of thumb:**

* `p < 0.10` → lift looks rare under placebo timing (more believable)
* `p ≥ 0.10` → lift is common under placebo timing (be cautious)

**Where to view:**

* Streamlit → Method Trust Checklist / Placebo Tests pages

---

## Common gotchas

* Bad pre-fit RMSE → SCM lift is not trustworthy (counterfactual is wrong)
* Short pre-period → unstable SCM weights + weak placebo selection
* Too granular donor grain → can overfit pre-period
* Very small true effects → estimates will look noisy (expected)

---

## “Finished” checklist (minimal hardening)

* `make demo` runs clean on a fresh clone
* `./scripts/demo.sh` works end-to-end
* `./scripts/smoke.sh` passes
* `make check` passes
* Streamlit runs without deprecation warnings (e.g., `use_container_width` removed)
* README documents: what it is, how to run, and how to interpret outputs

## Documentation
- [End-to-end guide (DOCX)](docs/end_to_end_guide.docx)

## Documentation
- docs/M5_Causal_Lift_End_to_End_Guide_v2.docx
