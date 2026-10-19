PY := .venv/bin/python
PYTHON ?= python3.13

.PHONY: help venv data enrich run cli scenarios test lint fmt all models gemini-models check verify-data

help:
	@echo "make venv    — создать .venv и установить зависимости"
	@echo "make data    — собрать POI из Overpass и построить data/pois.json"
	@echo "make enrich  — добавить описания из Wikipedia/Wikidata"
	@echo "make run     — запустить Streamlit-приложение"
	@echo "make cli     — пример запроса через CLI (demo-режим)"
	@echo "make scenarios — перегенерировать docs/TEST_SCENARIOS.md реальными прогонами"
	@echo "make test    — pytest (офлайн, без модели)"
	@echo "make lint    — ruff check"

venv:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m seruen.collect.overpass
	$(PY) -m seruen.collect.normalize

enrich:
	$(PY) -m seruen.collect.enrich_wiki

run:
	$(PY) -m streamlit run seruen/app/streamlit_app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false

cli:
	SERUEN_DEMO=1 $(PY) -m seruen.cli --preset mega --query "Я у Mega Silk Way, хочу погулять 45 минут"

scenarios:
	$(PY) tools/gen_test_scenarios.py

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check .

models:
	$(PY) tools/check_openai.py --list-models

gemini-models:
	$(PY) tools/check_gemini.py --list-models

check: test lint verify-data
	$(PY) tools/check_openai.py

verify-data:
	$(PY) -m seruen.collect.verify

fmt:
	.venv/bin/ruff check --fix .

all: data test lint
