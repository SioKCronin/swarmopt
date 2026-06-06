# SwarmOpt infrastructure

Build, run, release, and container tooling for the SwarmOpt repository.

| File | Purpose |
|------|---------|
| `Makefile` | `make init`, `make test`, `make release`, `make docker-gpu` |
| `requirements.txt` | Runtime deps for local dev / Docker (mirrors core library needs) |
| `requirements-dev.txt` | Extra dev and example deps (e.g. TDA tooling) |
| `run_tests.py` | Test suite entry point |
| `prepare_release.sh` | PyPI release preparation |
| `RELEASE_CHECKLIST.md` | Release checklist |
| `docker/` | GPU Docker image |
| `docker-compose.gpu.yml` | GPU demo compose stack |

## Local development

From the repository root:

```bash
pip install -e .
make -C infra init    # optional: example/TDA extras from requirements files
make -C infra test
```

Or run tools directly:

```bash
python infra/run_tests.py
./infra/prepare_release.sh
docker compose -f infra/docker-compose.gpu.yml up --build
```
