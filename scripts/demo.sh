set -euo pipefail
# Demo defaults (override via env vars)
CAMPAIGN_ID="${CAMPAIGN_ID:-cmp_demo}"
START_DATE="${START_DATE:-2014-08-01}"
END_DATE="${END_DATE:-2014-08-28}"
TREAT_FRAC="${TREAT_FRAC:-0.20}"
MAX_UPLIFT="${MAX_UPLIFT:-0.15}"
SEED="${SEED:-7}"

DONOR_GRAIN="${DONOR_GRAIN:-store_dept}"
USE_LOG1P="${USE_LOG1P:-1}"
ALPHA="${ALPHA:-50}"

N_PLACEBOS="${N_PLACEBOS:-50}"
MIN_PRE_DAYS="${MIN_PRE_DAYS:-28}"

PORT="${PORT:-8501}"

echo "==> Demo run"
echo "CAMPAIGN_ID=$CAMPAIGN_ID  WINDOW=$START_DATE..$END_DATE  TREAT_FRAC=$TREAT_FRAC  MAX_UPLIFT=$MAX_UPLIFT  SEED=$SEED"
echo "SCM: DONOR_GRAIN=$DONOR_GRAIN USE_LOG1P=$USE_LOG1P ALPHA=$ALPHA"
echo "Placebos: N_PLACEBOS=$N_PLACEBOS MIN_PRE_DAYS=$MIN_PRE_DAYS"
echo

# Ensure deps are installed
echo "==> make install"
make install

# End-to-end pipeline
echo "==> make pipeline"
make pipeline \
  CAMPAIGN_ID="$CAMPAIGN_ID" \
  START_DATE="$START_DATE" \
  END_DATE="$END_DATE" \
  TREAT_FRAC="$TREAT_FRAC" \
  MAX_UPLIFT="$MAX_UPLIFT" \
  SEED="$SEED" \
  DONOR_GRAIN="$DONOR_GRAIN" \
  USE_LOG1P="$USE_LOG1P" \
  ALPHA="$ALPHA"

# Placebo SCM
echo "==> make placebo"
make placebo \
  CAMPAIGN_ID="$CAMPAIGN_ID" \
  DONOR_GRAIN="$DONOR_GRAIN" \
  USE_LOG1P="$USE_LOG1P" \
  ALPHA="$ALPHA" \
  N_PLACEBOS="$N_PLACEBOS" \
  MIN_PRE_DAYS="$MIN_PRE_DAYS" \
  SEED="$SEED" \
  PROCESSED_DIR="data/processed"

echo
echo "==> Launching app on port $PORT"
echo "Open: http://localhost:$PORT"
make app PORT="$PORT"
