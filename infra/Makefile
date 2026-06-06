REPO_ROOT := $(abspath ..)

.PHONY: init test release docker-gpu

init:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	cd $(REPO_ROOT) && python infra/run_tests.py

release:
	cd $(REPO_ROOT) && ./infra/prepare_release.sh

docker-gpu:
	cd $(REPO_ROOT) && docker compose -f infra/docker-compose.gpu.yml up --build
