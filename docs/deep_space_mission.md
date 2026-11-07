# Deep-space autonomous swarm — research briefing

Status: **research prototype**. Nothing here is flight software, and none of
the tests described below are flight qualification.

## Mission concept

A distributed scientific swarm that operates autonomously for 20–30 years
without software updates. Each spacecraft observes its environment, ranks the
scientific value of possible measurements against a limited energy budget, and
keeps operating safely through long communication outages with Earth and with
its peers.

## Research hypothesis

Can a distributed scientific swarm maintain useful, safe, resource-efficient
autonomy for decades without maintenance despite communication delays,
incomplete observations and hardware degradation?

## Design rationale

Python SwarmOpt stays the home for research, benchmarks and reference
behaviour. Alongside it, `prototypes/deep_space/` is an independent Rust
execution core built for the constraints of long-lived embedded systems:

| Constraint | How the prototype addresses it |
|---|---|
| No heap fragmentation over decades | `no_std`, no allocator, all state in fixed-capacity arrays sized at compile time |
| Predictable worst-case work | PSO swarm size and dimension are const generics; iterations are capped; each decision costs exactly `P·(iterations+1)` objective evaluations |
| Replayable anomaly investigation | All randomness comes from a seeded SplitMix64 whose single-`u64` state is persisted; identical inputs give identical outputs |
| Optimizer bugs must not endanger the craft | The optimizer *proposes*; a separate safety gate re-derives energy use from the candidate list and approves or rejects |
| Corrupted inputs | Non-finite or out-of-range observations and objective values can never be selected or become a best solution |
| Hostile or noisy links | Strict, versioned, CRC-checked envelopes; sliding-window replay guard per peer |
| Reboots and torn writes | A/B snapshot slots with CRC and generation counter; sequence numbers jump a reserved gap on restore |
| Minimal supply chain | Zero third-party crates; `#![forbid(unsafe_code)]` |

Rust gives memory safety. It does **not**, by itself, give real-time
determinism, radiation tolerance, or qualification under any flight software
standard.

## Architecture

```
 Ground (Python SwarmOpt)          Flight research core (Rust, no_std)
 ────────────────────────          ───────────────────────────────────
 algorithm research                baseline ─┐
 mission simulation          ┌──── pso ──────┼──► planner ──► safety gate ──► execute
 benchmark comparison  ◄─────┤               │                    ▲
 (compare_python.py)         │     envelope ─┴── replay guard     │
                             │     state (A/B snapshots)          │
                             └──── adapters (future): sensors, comms, NVM,
                                   power, clocks, watchdog
```

The simulation in `sim.rs` stands in for the adapters at Stage 1.

## Prototype mission

Three spacecraft share a catalog of 64 targets. Science is renewable: every
50-tick epoch is a new observation window with fresh science values. Each
tick, a spacecraft:

1. recharges (capped at battery capacity);
2. receives due messages: decode → replay check → apply claim if current epoch;
3. sees up to 8 candidate observations, some possibly corrupted;
4. plans with either the **baseline** (best value-per-energy single pick) or
   the **PSO planner** (subset selection in `[0,1]^8`, penalized over budget);
5. passes the plan through the **safety gate**; on rejection, falls back to the
   baseline, and on a second rejection conserves energy;
6. executes, and broadcasts a claim for each observed target to its peers.

The light-time delay (base 2 ticks) means some duplicate observations happen
even on perfect links; faults make that worse, and the metrics show how much.

## Fault injection

`Faults::harsh()` enables everything at once: two link blackouts, periodic
drops, duplicates, single-bit corruption, extra delays, corrupted observations
(NaN, ±inf, negative), and a reboot of one craft with its newest snapshot
corrupted beforehand.

## Validation performed

See `prototypes/deep_space/README.md` for exact commands and the latest
results. In summary, the tests check:

- deterministic replay under nominal and harsh faults;
- energy reserve never violated (20 seeds × 2 strategies × 2 fault plans,
  plus a 16,000-tick harsh run);
- corrupted messages are rejected, not applied; every single-bit flip of an
  envelope is detected;
- duplicates are rejected; delayed out-of-order messages inside the window
  are accepted;
- corrupted observations are never executed;
- per-decision evaluation count equals its bound;
- total simulation state is fixed and under 64 KiB;
- warm restart, restart with corrupted latest snapshot (falls back to the
  older slot), and cold restart before any snapshot.

## Findings so far

- **The per-tick PSO planner does not beat the greedy baseline on science
  value** in this mission model (see README table). A likely cause is that a
  per-tick knapsack is myopic: it spends down to reserve on the current
  candidate set, which forecloses better-ratio targets next tick, while the
  ratio-greedy baseline is implicitly patient. That makes the next research
  question a multi-tick or receding-horizon objective, not a better per-tick
  optimizer.
- PSO plans are rejected by the safety gate a few percent of the time (the
  over-budget penalty is soft). The fallback path handles this, but it shows
  why the gate must be independent of the optimizer.
- PSO decisions cost roughly 60× more compute than the baseline (about 31 µs
  vs. 0.5 µs on a desktop), always within a fixed evaluation bound.

## Hardware roadmap

1. **Desktop deterministic simulation** (this PR).
2. **Long-duration fault injection**, including memory faults (bit flips in
   live state, not just in snapshots and messages) and clock faults.
3. **Commercial ARM / RISC-V boards** with measured power, memory and recovery
   behaviour. CI cross-builds the library for `thumbv7em-none-eabihf` and
   `riscv32imac-unknown-none-elf` to keep that path open.

None of these stages is flight qualification.

## Open questions

- Cold start: with no valid snapshot, the craft cannot know which sequence
  numbers it already used. A monotonic boot counter in separate NVM, or a
  per-boot sender nonce in the envelope, would close this gap.
- Envelopes are integrity-checked but not authenticated. CRC-32 catches
  accidents, not adversaries; an authenticated MAC is the next step if the
  threat model includes spoofing.
- Sequence numbers are `u32`; at one message per second that wraps in ~136
  years, so wrap handling is deferred rather than needed.
- How should the planner value information shared by peers whose claims
  arrive late (counted as `stale_claims`)?
