# SwarmOpt and DARPA DICE

Status: **research note, 2026-11-21.** This note maps SwarmOpt onto the problem in DARPA's *Decentralized AI through Controlled Emergence* (DICE, HR001126S0010) and lists the work that would bring the two closer. DICE itself no longer accepts new performers. It is used here as the clearest current statement of what a decentralized, adversary-tolerant collective has to show, and of how DARPA intends to measure it. Notes on the full Q&A are in Bureau at `docs/libraries/reference/darpa-dice-qa-notes.md`.

## Why the fit is close

DICE asks for **simple local rules that produce robust global behavior**, with no central orchestrator, sparse communication, partial observability, and agents that may be compromised. Particle swarm optimization started from exactly that idea. DARPA's Q&A confirms that physics-based local rules (Q10), stigmergic coordination through a shared environment (Q127), agents that select themselves into subtasks (Q98), and analytic agents mixed with foundation-model agents (Q113) are all in scope.

SwarmOpt already contains most of the pieces. What it lacks is the **framing and the measurements** DICE uses.

| DICE concept | What SwarmOpt has today | Gap |
| --- | --- | --- |
| Local rules → global behavior | `algo='local'` and `'unified'` neighborhoods (`k`, `u`); CPSO, multi-swarm | The local neighborhood is computed from **every** particle's position (`get_best_neighbor` sorts all N by distance). That is O(N²) per step and assumes global knowledge, which DICE rules out (Q73, Q125). |
| Sparse comms, partial observability | Deep-space core: claim/heartbeat envelopes between spacecraft, with communication faults | The Python library has no communication model. Every particle reads every other particle's state for free. |
| Robust to compromised agents | Strict, CRC-checked envelopes; per-peer replay guard; non-finite values can never become a best solution | Integrity checks only. Nothing yet stops a *well-formed but lying* agent from broadcasting a fake best position. |
| "Controlled" = bounded, statistically guaranteed | Respect boundary; optimizer proposes, separate safety gate approves (`safety.rs`) | No statistical statement about convergence or about keeping role coherence under perturbation. |
| Recovery after perturbation | Diversity monitoring and interventions; checkpoint/resume that survives power loss | Recovery isn't measured as DICE measures it: **messages needed to recover vs. N**. |
| Stigmergy | Deep-space sim: peers broadcast claims so others don't re-observe a target; the metrics count duplicated observations | It isn't named or measured as stigmergy, and claims are free assertions, not costly signals. |
| Heterogeneous agents | Many PSO variants that share one interface | All particles in a run use the same rule. |
| Evaluation at scale (500 / 5K / 100K agents) | Benchmarks and leaderboard on standard functions, with seeds and regret | Benchmarks measure cost on a single objective, not how the collective behaves as it grows or is attacked. |

## The research question SwarmOpt can own

> How much does a swarm have to say to stay coordinated, and how much can a lying member cost it?

This splits into two measurable claims, each of which fits DICE's framing:

1. **Recovery cost scales sub-quadratically.** With sparse, local, lossy communication, a swarm recovers from a perturbation (moving optimum, lost agents, a communication blackout) in O(N log N) or O(N) messages. Those are the Phase 2 and Phase 3 targets in DICE Q139.
2. **Costly signals beat free assertions under deception.** Q127 contrasts two ways of writing to a shared environment: a *sensing-coupled* write, where the signal is tied to work the agent did and costs it proportionally, and an *assertion* write, which is a free, unverified broadcast. Under Byzantine or deceptive members, the two should behave in opposite ways. A swarm whose "best position" broadcasts carry a check the receiver can verify cheaply (it re-evaluates the claimed point, or the claim is bound to an energy spend) should lose far less to liars than one that trusts reported costs. The safety gate's rule, "re-derive, never trust reported totals," is already the core of this idea.

## Work plan

### Track A: a communication-aware swarm in Python (the core)

1. **Neighborhoods with a comms graph.** Add a `topology` option (`ring`, `random-k`, `geometric radius`, `dynamic`) so each particle sees only the neighbors it can reach. Replace the all-pairs sort in `get_best_neighbor` with a lookup by neighbor list. Keep the current behavior as `topology='knn-oracle'` so old results stay reproducible.
2. **A message layer.** Particles exchange best-position messages over the graph, with configurable loss, delay, and bandwidth caps. Count every message. That count is the "interaction" unit DICE asks proposers to define (Q116, Q127).
3. **Adversary models.** Each run can include a fraction of compromised particles with one of these behaviors: crash, stale, noisy, and *liar* (reports a false best position or cost). Liars come in two variants, independent and colluding. This matches DICE's "colluding compromised agents" (Q120).
4. **Defenses.** Receivers spot-check claimed costs by evaluating the claimed point themselves, track neighbor reputation, and require a costly-signal mode in which a claim is bound to a recorded evaluation. Compare these with the current baseline, which trusts every claim.
5. **Dynamic objectives.** Use moving-peak and step-change versions of the existing benchmark functions so "time to recover" has a meaning.

### Track B: metrics in DICE's vocabulary

Add these to `benchmarks/metrics.py` and the leaderboard, next to simple regret:

- **Recovery messages vs. N**, fitted to a growth exponent and compared with O(N²), O(N log N), and O(N).
- **Time to recover**, in steps, after each perturbation.
- **Degradation under adversaries:** regret, plotted against the liar fraction, for each defense.
- **Coherence:** the share of particles still contributing to the group search, measured by the diversity metrics already in the library.
- **Total interactions per mission**, so runs line up with DICE's 5K / 50K / 1M interaction scales.

Scale targets: run 500 and 5,000 agents routinely and 100,000 at least once. Analytic PSO agents are cheap enough that 100K is realistic in vectorized NumPy or in the Rust core.

### Track C: the onboard core as the "controlled" half

The `no_std` deep-space core already demonstrates what DICE's TA2 asks for, a local control layer that keeps each agent safe regardless of what the group says:

- Write the safety gate up as **local inference control**. The optimizer proposes, the gate re-derives the quantities and decides, and peers can't override it.
- Extend envelopes with an optional **claim-evidence field** for the costly-signal mode, so Track A's defense also exists onboard.
- Add a fault-injection scenario with a lying peer to `tests/mission.rs` and report the duplicated-observation and energy cost with and without the defense.

### Track D (later): mixed collectives

DICE's agents are mainly foundation models, and DICE Q113 explicitly allows mixtures. One small experiment would cover this: a few LLM agents act as a scout layer that proposes regions, the PSO swarm exploits those regions, and the same liar models apply to the LLM agents. The analytic swarm then becomes a **cheap, verifiable substrate** for a foundation-model collective.

## How this feeds other projects

- Ocean-dynamics scenarios with blackout periods and drifting targets work as DICE-style *use cases*, and their configurations as *scenarios* (Q132).
- **AR learning lab:** "Watch a swarm survive a liar" is a strong interactive lesson. Start with free assertions, toggle on spot-checks, and the swarm visibly recovers. It fits the lab's theme of collective reasoning in animals, since honest signaling in animal groups is the biological version.
- **Fleet autonomy safety lab:** Tracks A–C are a self-contained first research result for the lab, and a credible preliminary result for a future DARPA proposal or for teaming with a DICE performer.
- **Publication:** Track A, plus the metrics and a JOSS paper on the library itself. The adversarial-recovery result is its own paper. Possible venues are ANTS (swarm intelligence), AAMAS, IEEE SIS, and *Swarm Intelligence* (Springer).

## Suggested first steps

1. Add `topology` and per-particle neighbor lists, with a test that `knn-oracle` reproduces the current results exactly.
2. Add the message counter and a loss/delay channel.
3. Add a liar adversary and the spot-check defense, then plot regret against liar fraction on sphere and Rastrigin at N = 500.
4. Add the recovery-messages-vs-N metric and run it at N = 500, 2,000, and 5,000 on a moving-peak function.

These four steps are enough for a first figure: how many messages recovery takes as the swarm grows, and how much regret liars cause with and without the defense. That figure is the core of the pitch.
