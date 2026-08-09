REPO_ROOT := $(abspath ..)
PYTHON ?= python3

.PHONY: init test release docker-gpu

init:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	cd $(REPO_ROOT) && $(PYTHON) infra/run_tests.py

release:
	cd $(REPO_ROOT) && ./infra/prepare_release.sh

docker-gpu:
	cd $(REPO_ROOT) && docker compose -f infra/docker-compose.gpu.yml up --build
