# Security Policy

## Supported versions

Security fixes go into the latest release on PyPI.

| Version | Supported |
| ------- | --------- |
| 0.3.x   | Yes       |
| < 0.3   | No        |

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
**Security → Report a vulnerability** on
[this repository](https://github.com/SioKCronin/swarmopt/security/advisories/new).
Do not open a public issue.

Include what you found, how to reproduce it, and the impact you expect.
You'll get an acknowledgement within 7 days and a plan or fix within 30 days
for confirmed issues. Credit is given in the advisory unless you ask otherwise.

## Verifying a release

Releases are built and published from GitHub Actions with PyPI trusted
publishing. Each distribution carries a build-provenance attestation you can
check with the GitHub CLI:

```bash
pip download swarmopt --no-deps -d dist/
gh attestation verify dist/swarmopt-*.whl --repo SioKCronin/swarmopt
```
