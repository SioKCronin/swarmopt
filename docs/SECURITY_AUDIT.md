# Security and data-integrity audit

**Scope:** SwarmOpt 0.3.0 (`26ba07e`): the library, benchmark and test
tooling, CI/CD, and everything that reads or writes data.
**Date:** 2026-11-18.
**Outcome:** fixes and the host persistence crate `rust/swarmopt-store`.
That crate is the host half of the Rust split; the onboard research core is
separate. See [`RUST_ARCHITECTURE.md`](RUST_ARCHITECTURE.md).

## Motivating case and threat model

The design target is a swarm of research drones under the ice of Ganymede,
optimising survey paths, sampling sites or formation. The same properties
matter on any unattended field hardware.

| Hazard | Consequence for stored data |
| --- | --- |
| Power loss at any instant | Torn writes, half-written files, lost progress |
| Jupiter's radiation belts (single-event upsets) | Bit flips in storage and RAM; damaged lengths can cause crashes or huge allocations |
| No trusted clock | Timestamp-based ordering or file names are unreliable |
| Fixed storage, intermittent uplink | Data must be bounded and pruned only after the surface has it |
| Several processes on one vehicle | Interleaved writes corrupt files |
| Data relayed between vehicles and to ground | Files cross trust boundaries; deserialising them must never run code |
| Decisions made on persisted results | A wrong "best position" means steering to the wrong place |

Accidental corruption is in scope. An active attacker who can rewrite
storage is only partly in scope (see *Residual risks*).

## Summary

| # | Finding | Severity | Status |
| --- | --- | --- | --- |
| 1 | `best_pos` didn't match `best_cost` (array aliasing) | **High** (integrity) | Fixed + regression test |
| 2 | Benchmarks recorded a seed that didn't reproduce the run | **High** (provenance) | Fixed |
| 3 | No crash safety in any file output | Medium | Fixed (atomic writes); Rust store for durable data |
| 4 | Same-second runs silently overwrote results | Medium | Fixed (exclusive, unique names) |
| 5 | NaN cost could take rank 1 on the leaderboard | Medium | Fixed |
| 6 | CSV formula injection in exported results | Low | Fixed |
| 7 | Non-standard JSON (`NaN`, `Infinity`) emitted | Low | Fixed |
| 8 | `intro_html` raw-HTML parameter undocumented | Low | Documented; image URI now escaped |
| 9 | `generate_scores.py` crashed on every invocation | Low (availability) | Fixed |
| 10 | No checkpoint/resume: a power cut lost the whole run | Medium (for the drone case) | New: `Swarm.optimize(checkpoint=...)` |
| — | Supply chain (CI, release) | — | Already strong; Rust job added |

### 1. Reported best position didn't score the reported best cost — High

`Particle.__init__` did `self.pos = self.best_pos = self.local_best_pos = rng.uniform(...)`,
so all three names pointed at one array. `update()` then does
`self.pos += self.velocity` **in place**, which dragged `best_pos` along
with the particle until the particle happened to improve. `Swarm.best_pos`
held references to particle arrays too.

Measured on 60 seeded runs (global/local/unified × 20 seeds): **6 reported a
`best_pos` whose cost wasn't `best_cost`**. For example, unified seed 10
reported 0.909 but `best_pos` actually scores 7.86.

Two related pairing bugs:

- `get_best_neighbor` paired each neighbour's *best cost* with its
  *current* position.
- The `sa` branch updated `best_cost` without `best_pos`.

**Fix:** independent copies at every assignment; neighbours report
`best_pos` with `best_cost`; the `sa` branch updates both.
`tests/unit/test_best_position_integrity.py` checks the invariant across
six algorithms × two functions × eight seeds, plus checks that no arrays
share memory.

*Impact:* results from 0.3.0 that used `best_pos` may be wrong. Numerical
results will change (they are now correct). Worth a note in the release.

### 2. Benchmark seeds were false provenance — High

`benchmarks/run_benchmarks.py` and `tests/benchmarks/run_suite.py` seeded
`np.random` / `random` globally and wrote that seed into the CSV. Since
0.3.0, `Swarm` draws only from its own `Generator` and ignores global state.
So the recorded seed had no effect: two runs both recorded `seed_base=0`
and produced different costs (1.100 vs 1.891).

**Fix:** pass `seed=` to `Swarm`; remove the global seeding. Two runs now
produce identical CSVs.

### 3–4. Non-atomic writes and silent overwrites — Medium

Every output (benchmark CSVs, legacy CSV, suite CSV/JSON, leaderboard HTML)
was opened with `"w"` and written in place. A crash left a truncated file
that the leaderboard then parsed. File names used one-second timestamps, so
two runs in the same second clobbered each other.

**Fix:** `swarmopt/_safe_io.py` writes to a temp file, fsyncs, renames and
fsyncs the directory. Timestamped outputs use `overwrite=False` (via
`link()`, which fails rather than replacing) plus `unique_path`.

### 5. NaN in a results CSV could win the leaderboard — Medium

`float("nan")` parses fine, and `nan < x` is always `False`. So a NaN row
seen first for a (function, algo) pair could never be replaced as "best",
and it sorted unpredictably. **Fix:** `finite_float` rejects NaN/±inf rows
in `leaderboard.py` and `generate_scores.py`.

### 6. CSV formula injection — Low

Algorithm and function names come from config JSON and end up in CSVs that
people open in spreadsheets. A name like `=HYPERLINK(...)` would execute
there (CWE-1236). **Fix:** `csv_safe` prefixes a `'` to string cells that
start with `= + - @ \t \r` and aren't numbers.

### 7. Non-standard JSON — Low

`json.dump` emitted `Infinity`/`NaN` for diverged runs, which many tools
reject. **Fix:** non-finite values become `null`, with `allow_nan=False`.

### 8. Raw HTML parameter — Low

`build_example_html_page(intro_html=...)` inserts markup unescaped by
design, but nothing said so. **Fix:** the docstring says it must be trusted,
author-written markup. `image_data_uri` is now attribute-escaped. All other
fields were already escaped, as was `benchmarks/leaderboard.py`.

### 9. `generate_scores.py` — Low

Missing imports (`os`, `csv`, `np`) and a call to a removed
`run_all_tests()`. It now delegates to `run_benchmarks.main`, and
`--leaderboard` works.

### 10. No checkpoint/resume — Medium for the drone case

A power cut lost the entire optimisation. See *New persistence layer*.

## Areas reviewed with no findings

- **Deserialisation:** no `pickle`, `eval`, `exec`, `yaml.load`, `joblib`
  or `torch.load` anywhere. The new payload encoding keeps it that way:
  strict JSON, a numeric-dtype allowlist, size and shape checks.
- **Subprocess:** `tests/index.py` and `tests/examples/index.py` use
  list-form `subprocess.run` without `shell=True`; the paths are
  repo-internal.
- **Network:** none in the library.
- **CI/CD:** actions pinned by SHA, `persist-credentials: false`,
  least-privilege `permissions`, no `pull_request_target`, PyPI trusted
  publishing (no stored tokens), a tag/version check, build-provenance
  attestations, Scorecard and Dependabot. The new `store.yml` follows the
  same rules and adds Cargo to Dependabot.
- **Config loading** (`run_suite.load_config`): reads a user-supplied local
  path. That's acceptable for a CLI, though note that a relative name falls
  back to the working directory.

## New persistence layer (`rust/swarmopt-store`)

The durable host data paths are implemented in `rust/swarmopt-store`, a
`std`-only, dependency-free core with PyO3 bindings. This is not the onboard
research core (`prototypes/deep_space`); see
[`RUST_ARCHITECTURE.md`](RUST_ARCHITECTURE.md). Full format spec:
`rust/swarmopt-store/README.md`.

- **Run log:** append-only and segmented. CRC-32C covers each record header
  (checked before the length is trusted) and each body. A sync marker lets
  readers skip past damage. Records are written to N independent replica
  directories and merged on read. Ordering is by sequence number only.
  There is a per-replica byte budget with `reject` (default) or
  `drop_oldest`, `prune_through(seq)` after downlink, a `stream_id` per
  mission, an exclusive lock, symlink refusal, and `0600`/`0700` permissions.
- **Checkpoints:** A/B slots written via temp file, fsync, rename and dir
  fsync. Generation counters, not clocks. On load, falls back to the
  previous generation and reports it.
- **Recovery on open:** torn tails are truncated; segments with damaged
  headers are quarantined (never deleted) and their sequence numbers are
  still counted, so numbers are never reused; empty half-created segments
  are removed.
- **`Swarm.optimize(checkpoint=..., checkpoint_every=N)`:** stores the full
  state (particles, swarm, RNG state) and resumes so that the final result
  is **bit-for-bit identical** to an uninterrupted run. A fingerprint of the
  configuration refuses to resume a different run.

### Verification

- Rust: 29 tests. These include detecting every single-bit flip in a record
  and a checkpoint; a power cut at every byte of a record and during segment
  creation; random bit flips in one replica (always fully repaired) and in
  both (a damaged record is never returned); 200 garbage files posing as
  segments (no panics); budgets; locking; stream mismatch; symlinks;
  permissions. Release builds keep overflow checks on.
- Python: a simulated power cut at three points × nine algorithm/operator
  variants resumes bit-exact. The test also asserts that the resumed run did
  fewer evaluations than a full run, which proves it continued from the
  checkpoint rather than restarting. Also tested: fallback after a bit flip
  in the newest checkpoint, configuration mismatch, hostile payloads
  (object dtypes, oversized shapes, NaN tokens), storage-full and lock
  errors.

## Residual risks and next steps

1. **Replication repairs, CRC only detects.** A record damaged in every
   replica is lost (reported in `missing_seqs`). For single-copy storage,
   add per-segment forward error correction (e.g. Reed-Solomon).
2. **No background scrubber.** Repairs happen in memory on read. Add a
   scrub that rewrites damaged replica segments from good copies.
3. **Integrity, not authenticity.** CRC stops accidents, not tampering. If
   relayed or downlinked data could be altered, add an HMAC (keyed per
   vehicle) or signatures at the payload or transport layer.
4. **Advisory locking.** It stops cooperating processes, not a hostile
   process running as the same user.
5. **Checkpoint coverage.** `cpso`, `hhoa`, multiobjective, PPSO and
   diversity-monitoring runs raise `NotImplementedError` with a checkpoint.
6. **RAM upsets during compute** aren't covered by storage CRCs. For flight
   use, consider running the optimiser redundantly and voting, or keep
   ECC memory and periodically check `best_pos` against `best_cost`. That
   invariant is now cheap to check and holds by construction.
7. **Windows** isn't covered by the new CI job yet (directory fsync is a
   no-op there).
