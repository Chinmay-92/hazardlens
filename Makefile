PY ?= .venv/bin/python

.PHONY: setup test eval eval-heldout run docker synth
setup:
	python3.13 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt
test:
	$(PY) -m pytest
eval:
	$(PY) scripts/evaluate.py
eval-heldout:
	$(PY) scripts/evaluate.py --data eval/heldout/images --out eval/heldout --seed 99 --regenerate
synth:
	$(PY) -c "from pathlib import Path; from hazardlens.synth import generate_dataset; generate_dataset(Path('data/samples'))"
run:
	$(PY) -m uvicorn hazardlens.app:app --reload --port 8080
docker:
	docker buildx build --platform linux/arm64 -t hazardlens:local --load .
