#!/usr/bin/env bash
set -euo pipefail

make pipeline CAMPAIGN_ID=cmp_smoke START_DATE=2014-08-01 END_DATE=2014-08-28 TREAT_FRAC=0.2 MAX_UPLIFT=0.15 SEED=7
make placebo CAMPAIGN_ID=cmp_smoke DONOR_GRAIN=store_dept USE_LOG1P=1 ALPHA=50 N_PLACEBOS=20

echo "Smoke test OK"
