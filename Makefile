.PHONY: help setup setup-core db index smoke smoke-free compare clean-db clean-cache

VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

help:
	@echo "  make setup       venv + every dependency (core, local embeddings, db, openai)"
	@echo "  make setup-core  venv + core only -- examples 01-05, nothing else needed"
	@echo "  make db          start Postgres 17 + pgvector on :5433, create the schema"
	@echo "  make index       chunk, embed and load the corpus"
	@echo "  make smoke       run all 38 examples and report"
	@echo "  make smoke-free  run them as a reader with no database would"
	@echo "  make compare     the chunking benchmark behind the README tables"
	@echo "  make clean-db    stop Postgres and delete its volume"
	@echo "  make clean-cache drop the embedding and LLM response caches"

$(PY):
	python3 -m venv $(VENV)

setup-core: $(PY)
	$(PIP) install -r requirements.txt

setup: setup-core
	$(PIP) install -r requirements-local.txt -r requirements-db.txt -r requirements-openai.txt
	@test -f .env || (cp .env.example .env && echo "wrote .env from .env.example")

db:
	docker compose up -d --wait
	$(PY) scripts/init_db.py

index:
	$(PY) scripts/index_corpus.py

smoke:
	$(PY) scripts/smoke_test.py

smoke-free:
	$(PY) scripts/smoke_test.py --no-db

compare:
	$(PY) examples/15_compare_chunking.py

clean-db:
	docker compose down -v

clean-cache:
	rm -rf .cache
