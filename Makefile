.PHONY: install lint test train evaluate serve gradio streamlit audit ci docker-build docker-run

PY := venv/Scripts/python
PIP := venv/Scripts/pip

# Unfixable CVEs (no fix version published as of 2026-05-20). See MANUAL_TASKS.md.
PIP_AUDIT_IGNORE := \
  --ignore-vuln PYSEC-2025-210 \
  --ignore-vuln PYSEC-2025-194 \
  --ignore-vuln PYSEC-2025-196 \
  --ignore-vuln PYSEC-2025-195 \
  --ignore-vuln PYSEC-2025-193 \
  --ignore-vuln PYSEC-2025-192 \
  --ignore-vuln PYSEC-2026-139 \
  --ignore-vuln PYSEC-2025-191 \
  --ignore-vuln PYSEC-2025-197 \
  --ignore-vuln PYSEC-2025-189 \
  --ignore-vuln PYSEC-2025-190 \
  --ignore-vuln PYSEC-2024-274 \
  --ignore-vuln PYSEC-2024-271 \
  --ignore-vuln PYSEC-2024-277

install:
	$(PIP) install -U pip wheel
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pre_commit install

lint:
	$(PY) -m black src/ tests/
	$(PY) -m isort src/ tests/ --profile black
	$(PY) -m flake8 src/ tests/
	$(PY) -m mypy src/

test:
	$(PY) -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70

train:
	$(PY) -m src.model.train --config config/config.yaml

evaluate:
	$(PY) -m scripts.run_evaluation --config config/config.yaml

serve:
	$(PY) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

gradio:
	$(PY) -m hf_space.app

streamlit:
	$(PY) -m streamlit run src/dashboard/streamlit_app.py

audit:
	$(PY) -m pip_audit -r requirements.txt $(PIP_AUDIT_IGNORE)
	$(PY) -m detect_secrets scan --baseline .secrets.baseline
	$(PY) -m bandit -r src/ -ll -ii

ci:
	$(PY) -m black --check src/ tests/
	$(PY) -m isort --check-only src/ tests/ --profile black
	$(PY) -m flake8 src/ tests/
	$(PY) -m mypy src/
	$(PY) -m bandit -r src/ -ll -ii
	$(PY) -m radon cc src/ -nc
	$(PY) -m interrogate src/ --fail-under=80
	$(PY) -m pip_audit -r requirements.txt $(PIP_AUDIT_IGNORE)
	$(PY) -m detect_secrets scan --baseline .secrets.baseline
	$(PY) -m pytest tests/ -v --tb=short --cov=src --cov-fail-under=70
	@echo "All CI gates green. Safe to git push."

docker-build:
	docker build -t before-it-breaks:latest .

docker-run:
	docker run --rm -p 8000:8000 before-it-breaks:latest
