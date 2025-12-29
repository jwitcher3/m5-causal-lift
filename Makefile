SHELL := /bin/bash

PY := .venv/bin/python
PIP := .venv/bin/pip

RAW_DIR ?= data/raw
GRAIN ?= store_dept
CAMPAIGN_ID ?= cmp_001
START_DATE ?= 2014-06-01
END_DATE ?= 2014-06-28
TREAT_FRAC ?= 0.2
MAX_UPLIFT ?= 0.15
SEED ?= 7
PROCESSED_DIR ?= data/processed
OUT_DIR := $(PROCESSED_DIR)


DONOR_GRAIN ?= store_dept
ALPHA ?= 50
USE_LOG1P ?= 1

PORT ?= 8501

.PHONY: help venv install dirs processed features simulate did scm eval placebo check pipeline app clean demo


help:
	@echo "Targets: venv install processed features simulate did scm eval pipeline app clean"

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	@if [ -f requirements.txt ]; then \
	$(PIP) install -r requirements.txt; \
	else \
	$(PIP) install polars numpy pandas streamlit altair statsmodels scipy; \
	fi

dirs:
	mkdir -p $(PROCESSED_DIR)

processed: dirs
	$(PY) src/m5lift/io/build_processed.py --raw_dir $(RAW_DIR) --out_dir $(OUT_DIR) --grain $(GRAIN)

features: dirs
	$(PY) src/m5lift/io/build_features.py --processed_dir $(OUT_DIR) --grain $(GRAIN)

simulate: dirs
	$(PY) src/m5lift/sim/simulator.py --processed_dir $(OUT_DIR) --grain $(GRAIN) \
	--campaign_id $(CAMPAIGN_ID) --start_date $(START_DATE) --end_date $(END_DATE) \
	--treat_frac $(TREAT_FRAC) --max_uplift $(MAX_UPLIFT) --seed $(SEED)

N_PLACEBOS ?= 50
MIN_PRE_DAYS ?= 28

placebo:
	$(PY) src/m5lift/methods/placebo_scm.py \
		--processed_dir $(PROCESSED_DIR) \
		--campaign_id $(CAMPAIGN_ID) \
		--donor_grain $(DONOR_GRAIN) \
		$(if $(filter 1,$(USE_LOG1P)),--use_log1p,) \
		--alpha $(ALPHA) \
		--n_placebos $(N_PLACEBOS) \
		--min_pre_days $(MIN_PRE_DAYS) \
		--seed $(SEED)

did: dirs
	$(PY) src/m5lift/methods/did_event_study.py --processed_dir $(OUT_DIR) --grain $(GRAIN) --campaign_id $(CAMPAIGN_ID)

scm: dirs
	@if [ "$(USE_LOG1P)" = "1" ]; then \
	$(PY) src/m5lift/methods/synth_control.py --processed_dir $(OUT_DIR) --campaign_id $(CAMPAIGN_ID) --donor_grain $(DONOR_GRAIN) --alpha $(ALPHA) --use_log1p; \
	else \
	$(PY) src/m5lift/methods/synth_control.py --processed_dir $(OUT_DIR) --campaign_id $(CAMPAIGN_ID) --donor_grain $(DONOR_GRAIN) --alpha $(ALPHA); \
	fi

eval: dirs
	$(PY) src/m5lift/eval/evaluate.py --processed_dir $(OUT_DIR)

check:
	@test -f $(PROCESSED_DIR)/fact_ground_truth.parquet || (echo "Missing fact_ground_truth.parquet (run: make simulate)"; exit 1)
	@test -f $(PROCESSED_DIR)/fact_method_results.parquet || (echo "Missing fact_method_results.parquet (run: make did/make scm)"; exit 1)
	@test -f $(PROCESSED_DIR)/fact_method_eval.parquet || (echo "Missing fact_method_eval.parquet (run: make eval)"; exit 1)
	@echo "OK: core artifacts exist."

pipeline: processed features simulate did scm eval placebo check

app:
	$(PY) -m streamlit run app/Home.py --server.port $(PORT)

clean:
	rm -rf $(PROCESSED_DIR)

demo:
	$(MAKE) clean
	$(MAKE) pipeline CAMPAIGN_ID=cmp_demo START_DATE=2014-08-01 END_DATE=2014-08-28 TREAT_FRAC=0.2 MAX_UPLIFT=0.15 SEED=7

 