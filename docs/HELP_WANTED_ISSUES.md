# Help Wanted Issues for SwarmOpt

Copy each section below into a new [GitHub Issue](https://github.com/SioKCronin/SwarmOpt/issues/new). Add labels such as `help wanted`, `good first issue`, or `enhancement` as suggested.

---

## Issue 1: Add `get_function(name)` to swarmopt.functions

**Labels:** `help wanted` `enhancement` `good first issue`

**Title:** Add `get_function(name)` to swarmopt.functions for lookup by string

**Body:**

### Context
`swarmopt.functions` has 30+ benchmark functions and `FUNCTION_METADATA` with names, bounds, and types. There is no single public API to get a function callable by its string name. The benchmark suite in `tests/benchmarks/run_suite.py` maintains its own `BENCHMARK_FUNCTIONS` map and a local `get_function_by_name()`.

### Task
- Add a public function in `swarmopt/functions.py`, e.g. `get_function(name: str)` (or `get_function_by_name`), that returns the callable for a given function name (e.g. `"sphere"`, `"ackley"`).
- Use the existing `FUNCTION_METADATA` keys and the module’s own function definitions so the list of names stays in one place.
- Optionally add a small `__all__` or docstring note listing the supported names, or add a helper like `list_function_names()` that returns the available names.

### Acceptance criteria
- [ ] `from swarmopt.functions import get_function` works.
- [ ] `get_function("sphere")` returns the sphere callable; `get_function("unknown")` returns `None` or raises a clear error.
- [ ] `tests/benchmarks/run_suite.py` is updated to use `get_function()` (and can drop or shrink its local `BENCHMARK_FUNCTIONS` / `get_function_by_name` if desired).
- [ ] Any new helper is briefly documented in the module docstring or README.

### Hints
- Inspect `FUNCTION_METADATA.keys()` and `getattr(swarmopt.functions, name)` for valid names.
- Handle names that are not callable (e.g. `FUNCTION_METADATA`, `get_function_metadata`) so they are not returned.

---

## Issue 2: Extend researcher benchmark suite with all test functions

**Labels:** `help wanted` `enhancement` `good first issue`

**Title:** Extend benchmark suite to include all single-objective test functions

**Context**
`tests/benchmarks/run_suite.py` currently wires a subset of test functions (e.g. sphere, rosenbrock, ackley, rastrigin, griewank, weierstrass, sum_squares, zakharov, levy). `swarmopt.functions` defines 30+ functions with `FUNCTION_METADATA` (e.g. schwefel, michalewicz, perm, trid, beale, booth, hartmann_3d, hartmann_6d, shekel, etc.).

**Task**
- Once `get_function(name)` exists (see related issue), use it in the benchmark suite so that any function name from the library can be used in configs without maintaining a duplicate map.
- Alternatively (or in addition), extend the benchmark runner’s function list so that all single-objective functions in `FUNCTION_METADATA` that are implemented as callables in `swarmopt.functions` can be selected in config (e.g. via `"functions": ["sphere", "ackley", "schwefel", ...]` or a preset like `"all"`).
- Ensure bounds are taken from `get_function_metadata(name)` (or equivalent) so each function runs in its intended domain.

**Acceptance criteria**
- [ ] Benchmark configs can specify any supported single-objective function by name (no hardcoded subset required in the script).
- [ ] At least one built-in config (e.g. `full` or a new `all_functions`) runs over a large subset of the 30+ functions where feasible.
- [ ] `--list-functions` (or equivalent) reflects the full set of available functions.
- [ ] Docs in `tests/benchmarks/README.md` are updated to describe how to select functions.

---

## Issue 3: Add type hints to core public API

**Labels:** `help wanted` `enhancement` `documentation`

**Title:** Add type hints to Swarm, Particle, and main function signatures

**Context**
SwarmOpt is used by researchers and students. Type hints improve IDE support, readability, and enable optional static checking (e.g. mypy).

**Task**
- Add type annotations to the public API, starting with:
  - `swarmopt/swarm.py`: `Swarm.__init__` parameters and key attributes (e.g. `best_cost`, `best_pos`, `runtime`); `Particle` attributes and `update()` if public.
  - `swarmopt/functions.py`: at least the main benchmark function signatures, e.g. `def sphere(x: np.ndarray) -> float` (or `Union[float, np.ndarray]` where appropriate).
- Use `from __future__ import annotations` if helpful to avoid string quotes for forward references.
- No need to type private or internal helpers exhaustively; focus on what users and downstream tools see.

**Acceptance criteria**
- [ ] Core constructors and main methods have parameter and return types where it makes sense.
- [ ] Project can run under mypy (or pyright) with a relaxed config (e.g. no strict mode required) without errors on the modified files; or document “type hints added, mypy optional” in CONTRIBUTING/README.
- [ ] README or CONTRIBUTING mentions that type hints are used and optional static checking is welcome.

---

## Issue 4: Add API reference / parameter table to docs

**Labels:** `help wanted` `documentation` `good first issue`

**Title:** Add API reference: Swarm parameters, algorithms, and options

**Context**
New users need a single place to see all `Swarm` parameters, which algorithms exist (`algo=...`), and how inertia/velocity-clamping/variation options are named. Currently this is spread across README examples and code.

**Task**
- Add a doc (e.g. `docs/API.md` or a section in README) that includes:
  - A table of `Swarm.__init__` parameters: name, type, default, short description.
  - A list of supported `algo` values (global, local, unified, cpso, hhoa, etc.) with one-line descriptions.
  - Lists or tables for `inertia_func`, `velocity_clamp_func`, and (if public) variation/diversity options.
- Optionally add a table of single-objective benchmark functions (name, type unimodal/multimodal, bounds, optimal value) based on `FUNCTION_METADATA`.

**Acceptance criteria**
- [ ] One clearly named doc or section serves as the API/parameter reference.
- [ ] README or main docs link to it.
- [ ] Tables are generated from code or manually kept in sync (e.g. “see swarmopt.functions.FUNCTION_METADATA”).

---

## Issue 5: Add unit tests for velocity_clamping and variation utils

**Labels:** `help wanted` `testing` `good first issue`

**Title:** Add unit tests for velocity_clamping and variation modules

**Context**
Integration tests already cover velocity clamping and variation operators in full runs. Unit tests would lock down the behavior of individual helpers (e.g. that clamping functions return arrays of the right shape, that variation strategies don’t mutate inputs in place, boundary cases).

**Task**
- Add tests under `tests/unit/` for:
  - **Velocity clamping:** For each function in `swarmopt.utils.velocity_clamping` (e.g. basic_clamping, adaptive_clamping), call with sample inputs and check output shape, that values lie within expected bounds, and that edge cases (e.g. zero velocity, max iteration) are handled.
  - **Variation:** For selected strategies in `swarmopt.utils.variation` (e.g. gaussian_variation, uniform_variation, boundary_variation), call `apply_variation` with a known position and bounds; check output shape, that result stays within bounds when applicable, and that the original position is not mutated if that’s the intended contract.
- Reuse or import from `tests/unit/context.py` for imports; follow existing pytest style in `tests/unit/`.

**Acceptance criteria**
- [ ] New test file(s) in `tests/unit/` for velocity_clamping and variation.
- [ ] `python run_tests.py --unit` (or `pytest tests/unit/`) passes.
- [ ] At least one test per public clamping function and at least 2–3 variation strategies covered.

---

## Issue 6: Create a “Getting Started” Jupyter notebook

**Labels:** `help wanted` `documentation` `good first issue`

**Title:** Add a Getting Started Jupyter notebook for SwarmOpt

**Context**
Researchers and students often learn by running notebooks. A single “Getting started” notebook would complement the README and examples.

**Task**
- Add a Jupyter notebook (e.g. `tutorials/getting_started.ipynb` or `examples/Getting_Started.ipynb`) that:
  - Installs or imports swarmopt and runs a minimal PSO (e.g. sphere with global best).
  - Shows how to change one or two options (e.g. inertia_func, velocity_clamp_func or algo).
  - Optionally shows a simple convergence plot (best cost vs epoch) using matplotlib.
  - Points to the full test suite and benchmark runner for more advanced use.
- Keep it short (a few cells), runnable in a standard environment (numpy, swarmopt; matplotlib optional).

**Acceptance criteria**
- [ ] Notebook runs from top to bottom without errors on a clean env with swarmopt installed.
- [ ] README or docs link to the notebook (and mention that Jupyter is optional).
- [ ] Notebook is checked in under a clear path (e.g. `tutorials/` or `examples/`).

---

## Issue 7: Document all benchmark functions in one reference table

**Labels:** `help wanted` `documentation` `good first issue`

**Title:** Add a single reference table of all benchmark functions

**Context**
`swarmopt.functions` has 30+ single-objective functions with metadata (bounds, type, optimal value). This is useful for papers and for choosing functions in the benchmark suite. Right now the README lists only a subset.

**Task**
- Add one reference (e.g. in `docs/BENCHMARK_FUNCTIONS.md` or a section in README) that contains a table of all single-objective benchmark functions.
- Columns could be: **Name**, **Type** (unimodal/multimodal), **Bounds**, **Optimal value**, **Notes** (e.g. “dimension-dependent”).
- Prefer generating the table from `FUNCTION_METADATA` (e.g. a small script that prints Markdown or RST) so it stays in sync; otherwise document that the table is maintained manually from that dict.

**Acceptance criteria**
- [ ] Every function in `FUNCTION_METADATA` appears in the table (or is explicitly listed as “not for benchmarks” with a reason).
- [ ] README or main docs link to this reference.
- [ ] Table is either generated from code or explicitly documented as mirroring `FUNCTION_METADATA`.

---

## Issue 8: Run benchmark suite in CI (optional job)

**Labels:** `help wanted` `ci` `enhancement`

**Title:** Add optional CI job to run researcher benchmark suite

**Context**
The benchmark suite in `tests/benchmarks/run_suite.py` is useful for regression and for validating that algorithms still run. Running it in CI would catch breakage early.

**Task**
- Add a GitHub Actions job (or equivalent) that:
  - Runs the benchmark suite with a small config (e.g. `--config quick`) so the job stays short (e.g. under 5 minutes).
  - Optionally uploads the generated CSV/JSON as an artifact so maintainers can inspect results.
- The job can be optional (e.g. on push to main/release tags) or required; document the choice in the workflow or in CONTRIBUTING.

**Acceptance criteria**
- [ ] A new job (or step) runs `tests/benchmarks/run_suite.py --config quick` (or similar) and fails the run if the script exits non-zero.
- [ ] Dependencies (numpy, scipy if needed) are installed in the job.
- [ ] README or CONTRIBUTING mentions that CI runs the benchmark suite (and link to the workflow file).

---

## How to use this file

1. Open [GitHub Issues](https://github.com/SioKCronin/SwarmOpt/issues/new).
2. Copy the **Title** and **Body** of the issue you want.
3. Add the suggested **Labels** (create them in the repo if they don’t exist: `help wanted`, `good first issue`, `enhancement`, `documentation`, `testing`, `ci`).
4. Submit the issue.

You can create all of them at once or start with a few (e.g. 1, 2, 4, 6) that are most useful for your roadmap.
