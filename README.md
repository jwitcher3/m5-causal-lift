# M5 Causal Lift (Incrementality Sandbox)

An end-to-end **incrementality sandbox** on **M5-style simulated retail data**.  
It simulates treatment/control campaigns and evaluates causal methods against **known ground truth**.

## What you get
- **Campaign simulator** (treated/control + true effect)
- **Methods**
  - DiD / Event Study (TWFE-style)
  - Synthetic Control (ridge SCM; optional log1p fit)
- **Evaluation vs ground truth** (scale-aware scoring table)
- **Streamlit app** to explore results, method scorecards, SCM interpretability (donor weights), and trust checks (placebos)

---

## Quickstart

### 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
