# M5 Causal Lift (Incrementality Sandbox)
cat > README.md <<'MD'
# M5 Causal Lift (Incrementality Sandbox)

A small, end-to-end **incrementality sandbox** using **M5-style simulated retail data**.
It simulates treatment/control campaigns and evaluates causal methods against **known ground truth**.

## What you get
- Campaign simulator (creates treated/control + true effect)
- Methods:
  - DiD / event study
  - Synthetic Control (ridge SCM, optional `log1p` fit)
- Evaluation vs ground truth (scale-aware)
- Streamlit app to explore results + scorecard + diagnostics
- SCM interpretability: donor weights + lift stability (CV)

## Quickstart

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
