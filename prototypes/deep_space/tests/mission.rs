//! Mission-level tests: determinism, energy-reserve enforcement under every
//! fault class, communication loss, delayed/duplicate/corrupted messages,
//! corrupted observations, bounded runtime, restart and state recovery.

use deep_space::planner::PLANNER_PARTICLES;
use deep_space::sim::{self, Faults, SimConfig, Strategy, Window, MAX_BLACKOUTS};

const STRATEGIES: [Strategy; 2] = [Strategy::Baseline, Strategy::Pso];

fn cfg(seed: u64, strategy: Strategy, faults: Faults) -> SimConfig {
    let mut c = SimConfig::nominal(seed, strategy);
    c.faults = faults;
    c
}

#[test]
fn deterministic_replay() {
    for s in STRATEGIES {
        for f in [Faults::NONE, Faults::harsh()] {
            let c = cfg(11, s, f);
            assert_eq!(sim::run(&c), sim::run(&c), "{s:?}");
        }
    }
}

#[test]
fn different_seeds_differ() {
    let a = sim::run(&cfg(1, Strategy::Pso, Faults::NONE));
    let b = sim::run(&cfg(2, Strategy::Pso, Faults::NONE));
    assert_ne!(a, b);
}

#[test]
fn energy_reserve_never_violated() {
    for seed in 0..20 {
        for s in STRATEGIES {
            for f in [Faults::NONE, Faults::harsh()] {
                let m = sim::run(&cfg(seed, s, f));
                assert!(
                    m.min_reserve_margin >= 0.0,
                    "seed {seed} {s:?}: margin {}",
                    m.min_reserve_margin
                );
                assert!(m.unique_science.is_finite() && m.energy_used.is_finite());
            }
        }
    }
}

#[test]
fn reserve_above_capacity_means_always_conserve() {
    for s in STRATEGIES {
        let mut c = cfg(3, s, Faults::NONE);
        c.reserve = c.capacity + 1.0;
        let m = sim::run(&c);
        assert_eq!(m.observations, 0);
        assert_eq!(m.conserve_decisions, m.decisions);
        assert_eq!(m.energy_used, 0.0);
    }
}

#[test]
fn nominal_run_is_clean() {
    for s in STRATEGIES {
        let m = sim::run(&cfg(5, s, Faults::NONE));
        assert!(m.observations > 0);
        assert_eq!(m.rejected_decode, 0);
        assert_eq!(m.rejected_replay, 0);
        assert_eq!(m.messages_dropped, 0);
        assert_eq!(m.queue_overflow, 0);
        assert_eq!(m.messages_accepted, m.messages_delivered);
    }
}

#[test]
fn total_blackout_increases_duplicate_work() {
    let mut f = Faults::NONE;
    f.blackouts = [Some(Window {
        start: 0,
        end: u32::MAX,
    }); MAX_BLACKOUTS];
    for s in STRATEGIES {
        let ok = sim::run(&cfg(8, s, Faults::NONE));
        let dark = sim::run(&cfg(8, s, f));
        assert_eq!(dark.messages_accepted, 0);
        assert_eq!(dark.messages_dropped, dark.messages_sent);
        assert!(
            dark.duplicate_observations > ok.duplicate_observations,
            "{s:?}"
        );
        // Still safe and still doing science while isolated.
        assert!(dark.min_reserve_margin >= 0.0);
        assert!(dark.unique_science > 0.0);
    }
}

#[test]
fn corrupted_messages_are_rejected_not_applied() {
    let mut f = Faults::NONE;
    f.corrupt_every = 3;
    let m = sim::run(&cfg(4, Strategy::Baseline, f));
    let expected_corrupt = m.messages_sent / 3;
    assert_eq!(m.rejected_decode, expected_corrupt);
    assert_eq!(
        m.messages_accepted + m.rejected_decode,
        m.messages_delivered
    );
}

#[test]
fn duplicate_messages_are_rejected_by_replay_guard() {
    let mut f = Faults::NONE;
    f.duplicate_every = 2;
    let m = sim::run(&cfg(4, Strategy::Pso, f));
    assert_eq!(m.rejected_replay, m.messages_sent / 2);
    assert_eq!(m.rejected_decode, 0);
}

#[test]
fn delayed_messages_within_window_are_accepted() {
    let mut f = Faults::NONE;
    f.delay_every = 3;
    f.extra_delay = 4;
    let m = sim::run(&cfg(4, Strategy::Baseline, f));
    assert_eq!(m.rejected_replay, 0);
    assert_eq!(m.messages_accepted, m.messages_sent);
}

#[test]
fn corrupted_observations_never_executed() {
    let mut f = Faults::NONE;
    f.corrupt_observation_every = 2;
    for s in STRATEGIES {
        let m = sim::run(&cfg(6, s, f));
        assert!(m.corrupted_candidates > 0);
        assert!(m.unique_science.is_finite());
        assert!(m.min_reserve_margin >= 0.0);
    }
}

#[test]
fn bounded_decision_cost() {
    let c = cfg(9, Strategy::Pso, Faults::harsh());
    let m = sim::run(&c);
    let bound = PLANNER_PARTICLES as u32 * (c.planner_iterations + 1);
    assert_eq!(m.max_evals_per_decision, bound);
    assert!(m.total_evals <= bound * m.decisions);
}

#[test]
fn fixed_memory_footprint() {
    // Whole mission state, stack-allocated; no heap anywhere in the library.
    assert!(
        sim::state_bytes() < 64 * 1024,
        "{} bytes",
        sim::state_bytes()
    );
}

#[test]
fn restart_recovers_from_snapshot_and_peers_keep_accepting() {
    let mut f = Faults::NONE;
    f.restart = Some((0, 155));
    let m = sim::run(&cfg(12, Strategy::Pso, f));
    assert_eq!(m.restarts, 1);
    assert_eq!(m.cold_starts, 0);
    assert_eq!(m.snapshot_fallbacks, 0);
    // Sequence numbers resume above the reserved gap, so no message from the
    // rebooted craft is mistaken for a replay.
    assert_eq!(m.rejected_replay, 0);
    assert!(m.min_reserve_margin >= 0.0);
}

#[test]
fn restart_with_corrupted_latest_snapshot_falls_back() {
    let mut f = Faults::NONE;
    f.restart = Some((2, 155));
    f.corrupt_snapshot_before_restart = true;
    let m = sim::run(&cfg(12, Strategy::Baseline, f));
    assert_eq!(m.restarts, 1);
    assert_eq!(m.snapshot_fallbacks, 1);
    assert_eq!(m.cold_starts, 0);
    assert_eq!(m.rejected_replay, 0);
}

#[test]
fn restart_before_any_snapshot_cold_starts_safely() {
    let mut f = Faults::NONE;
    f.restart = Some((1, 0));
    let m = sim::run(&cfg(12, Strategy::Pso, f));
    assert_eq!(m.cold_starts, 1);
    assert!(m.min_reserve_margin >= 0.0);
}

#[test]
fn long_duration_harsh_run_stays_safe() {
    // ~50k decisions; a smoke test for long-run stability, not a 30-year proof.
    let mut c = cfg(2026, Strategy::Baseline, Faults::harsh());
    c.ticks = 16_000;
    let m = sim::run(&c);
    assert!(m.min_reserve_margin >= 0.0);
    assert_eq!(m.queue_overflow, 0);
}
