## What this project is

**M5 Causal Lift** is an incrementality sandbox that:
1) Simulates a campaign on top of the M5-style retail data generating process (DGP)
2) Produces **ground truth** causal lift (known ATT)
3) Runs causal methods (e.g., **DiD / TWFE**, **Synthetic Control (ridge)**)
4) Evaluates each method against the known truth and surfaces results in a Streamlit app

The goal is to build intuition and guardrails for incrementality measurement (fit quality, stability, pretrend risk),
using a fully controlled environment where "truth" is observable.

---

## Quickstart

Create data + run methods end-to-end:

```bash
make pipeline
make app


# M5 Causal Lift (Incrementality Sandbox)

End-to-end portfolio project using the M5 Forecasting dataset as a base, with simulated campaigns (ground-truth lift),
DiD/Event Study, and Synthetic Control, evaluated against the simulator.

## Quickstart

```bash
make help
make install
make pipeline
make app
'''
