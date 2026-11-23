![particles](https://github.com/SioKCronin/PSO-baselines/blob/master/media/swarmopt_lateral.png)

# SwarmOpt

SwarmOpt is a library of swarm optimization algorithms implemented in Python. 

Swarm intelligence leverages population-based search solutions to balance exploration and exploitation with respect 
to specified cost functions. The PSO lineage was sparked by Eberhart and Kennedy in their original paper on PSOs in 1995, 
and the intervening years have seen many variations spring from their central idea. 

## Installation

To install SwarmOpt, run this command in your terminal:

```shell
$ pip install swarmopt
```

## Quick Start

```python
from swarmopt import Swarm
from swarmopt.functions import sphere

# Basic usage
swarm = Swarm(
    n_particles=30,
    dims=2,
    c1=2.0,
    c2=2.0,
    w=0.9,
    epochs=100,
    obj_func=sphere,
    algo='global'
)

swarm.optimize()
print(f"Best cost: {swarm.best_cost}")
```

### Reproducible runs

Pass `seed` to get the same result every time. SwarmOpt draws all randomness
from its own `numpy.random.Generator` and never touches NumPy's or Python's
global random state.

### Neighbourhoods that do not see the whole swarm

`topology='knn-oracle'` is the default: every particle can read every other
particle. A communication topology gives each particle a neighbour list and
counts the claims those neighbours actually send.

```python
swarm = Swarm(n_particles=30, dims=2, c1=2.0, c2=2.0, w=0.9,
              epochs=100, obj_func=sphere, algo='local', seed=42,
              topology='ring', k=4, message_loss=0.1, message_delay=1,
              liar_fraction=0.2, defense='spot-check')
swarm.optimize()
print(swarm.messages_sent, swarm.best_cost)
```

`defense='spot-check'` re-evaluates a claimed point instead of trusting the
cost a neighbour reports. `knn-oracle` does not take loss, delay, or liars.

```python
swarm = Swarm(n_particles=30, dims=2, c1=2.0, c2=2.0, w=0.9,
              epochs=100, obj_func=sphere, seed=42)
```

### Surviving power cuts (checkpoint and resume)

For long runs on hardware that can lose power, checkpoints let a run pick up
where it stopped and finish exactly as if it had never been interrupted.
They need the Rust storage extension in
[`rust/persist`](rust/persist/) (`swarmopt-persist`):

```python
from swarmopt import Swarm, persist

with persist.SwarmCheckpoint("run/ckpt") as ckpt:
    swarm = Swarm(n_particles=30, dims=2, c1=2.0, c2=2.0, w=0.9,
                  epochs=1000, obj_func=sphere, seed=42)
    swarm.optimize(checkpoint=ckpt, checkpoint_every=25)  # resumes if a checkpoint exists
```

See [`docs/SECURITY_AUDIT.md`](docs/SECURITY_AUDIT.md) for the threat model
behind it (unattended drones under the ice of Ganymede). How this crate sits
next to the onboard research core is in
[`docs/RUST_ARCHITECTURE.md`](docs/RUST_ARCHITECTURE.md).

## Testing

Run the comprehensive test suite:

```bash
# Unit tests (what CI runs on every pull request)
python -m pytest

# Run all tests, including the slower integration scripts
python infra/run_tests.py

# Or run specific test categories
python infra/run_tests.py --unit           # Fast unit tests only
python infra/run_tests.py --show           # Show all available tests
python tests/index.py                      # Interactive test runner
```

Build, release, Docker, and dependency pin files live under [`infra/`](infra/).

For local development from a clone:

```bash
pip install -e ".[dev]"
make -C infra init   # optional extras (matplotlib, TDA examples, etc.)
```

## Benchmarks

Stratified empirical comparisons (repeated trials, spread, ranks per function class) are in [`benchmarks/README.md`](benchmarks/README.md). Run the driver from the repo root with `python benchmarks/run_benchmarks.py` (see that file for options and requirements).

## Algorithms

### Single-Objective
* Global Best PSO - Kennedy & Eberhart 1995
* Local Best PSO - Kennedy & Eberhart 1995
* Unified PSO - Parsopoulos &  Vrahatis 2004
* Dynamic Multi-Swarm PSO - Liang & Suganthan 2005 *(being reworked; not available in 0.3)*
* Simulated Annealing PSO - Mu, Cao, & Wang 2009
* Cooperative PSO (CPSO) - Van den Bergh & Engelbrecht 2004
* Horse Herd Optimization Algorithm (HHOA) - Ibrahim et al. 2025

### Multiobjective
* Multiobjective PSO - Handles multiple conflicting objectives simultaneously 

## Every algorithm, one interface

`Swarm` runs the core PSO family above and 55 more nature-inspired optimizers from `swarmopt.nio`: grey wolf, firefly, ant colony, differential evolution, water cycle and the rest. Name one as `algo` and it searches the same space, with the same seed giving the same run:

```python
import swarmopt
from swarmopt import Swarm
from swarmopt.functions import rastrigin

swarmopt.algorithms()                  # every algo name, with its family
swarmopt.algorithms("evolutionary")    # just one family

swarm = Swarm(n_particles=30, dims=5, c1=2.0, c2=2.0, w=0.9, epochs=200,
              obj_func=rastrigin, algo="gwo", velocity_clamp=(-5.12, 5.12), seed=1)
swarm.optimize()
print(swarm.best_cost, swarm.best_pos)
```

What carries across:

* **Reproducibility.** The same seed (an int, `SeedSequence` or `Generator`) gives the same run.
* **Checkpoint and resume.** `swarm.optimize(checkpoint=..., checkpoint_every=N)` works for every nio algorithm, and a resumed run finishes exactly as an uninterrupted one.
* **Per-algorithm settings.** Pass them as `algo_params`, for example `algo_params={"num_rivers": 4}` for `wca`.

What doesn't yet: communication topologies (message loss, lying peers, spot-check), `respect_boundary`, diversity monitoring and variation operators are PSO-family features. `Swarm` refuses them for nio algorithms rather than silently running without them. Bringing the communication layer to every algorithm is the next piece of work.

`pso` and `ppso` run the core global-best PSO and proactive PSO. The standard-library versions in `swarmopt.nio` remain for the RL environment and for use without NumPy.

Each optimizer is grouped by the metaphor it comes from (`swarm`, `evolutionary`, `ecological`, `physical`). A metaphor is not a mechanism, and some metaphor-based methods reduce to familiar operators (Sörensen, 2015, ["Metaheuristics—the metaphor exposed"](https://doi.org/10.1111/itor.12001)). So every method can be compared at an equal number of objective evaluations: see [docs/nio_benchmarks.md](docs/nio_benchmarks.md) and `swarmopt.nio.compare`.

`swarmopt.nio` also has the classic collective-motion models (Boids and Vicsek), grid pathfinders, and a small reinforcement-learning training environment, and uses only the Python standard library. [docs/nio.md](docs/nio.md) lists every algorithm with a free-to-read reference where one exists.

## Benchmark Functions

### Single Objective
* Sphere Function
* Rosenbrock's Function
* Ackley's Function
* Griewank's Function
* Rastrigin's Function
* Weierstrass Function

### Multiobjective
* **ZDT1, ZDT2, ZDT3** - Zitzler-Deb-Thiele test functions
* **DTLZ1, DTLZ2** - Deb-Thiele-Laumanns-Zitzler test functions

## Implemented Features

### Inertia Weight Variations
* **Constant** - Traditional fixed inertia weight
* **Linear Decreasing** - Classic linear decay (default)
* **Chaotic** - Chaotic inertia using logistic map
* **Random** - Random inertia between 0.5-1.0
* **Adaptive** - Adapts based on convergence progress
* **Chaotic-Random** - Combination of chaotic and random
* **Exponential Decreasing** - Exponential decay
* **Sigmoid Decreasing** - Sigmoid decay curve

### Velocity Clamping Variations
* **No Clamping** - Particles can move freely
* **Basic Clamping** - Standard velocity bounds
* **Adaptive Clamping** - Decreases over time
* **Exponential Clamping** - Exponential decay
* **Sigmoid Clamping** - Sigmoid decay
* **Random Clamping** - Random bounds
* **Chaotic Clamping** - Chaotic bounds using logistic map
* **Soft Clamping** - Soft bounds using tanh
* **Hybrid Clamping** - Adaptive + exponential
* **Convergence-Based** - Based on optimization progress

## Sample Applications

* Neural network number of layers and weight optimization
* Routing in communication networks
* Satellite repair helper-swarm standoff positioning

## Security

See [SECURITY.md](SECURITY.md) to report a vulnerability privately or to
verify a release's build provenance.

## Citation

Siobhan K Cronin, SwarmOpt (2018), GitHub repository, https://github.com/SioKCronin/SwarmOpt
