# SwarmOpt infrastructure

Build, run, release, and container tooling for the SwarmOpt repository.

| File | Purpose |
|------|---------|
| `Makefile` | `make init`, `make test`, `make release`, `make docker-gpu` |
| `run_tests.py` | Test suite entry point |
| `prepare_release.sh` | PyPI release preparation |
| `RELEASE_CHECKLIST.md` | Release checklist |
| `docker/` | GPU Docker image |
| `docker-compose.gpu.yml` | GPU demo compose stack |

Run from the repository root:

```bash
python infra/run_tests.py
make -C infra test
./infra/prepare_release.sh
docker compose -f infra/docker-compose.gpu.yml up --build
```
