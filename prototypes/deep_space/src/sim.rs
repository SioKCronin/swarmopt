//! Three-spacecraft mission simulation with deterministic fault injection.
//!
//! Each tick, every spacecraft recharges, receives any messages due, sees up
//! to [`MAX_CANDIDATES`] observation opportunities from a shared catalog of
//! [`N_TARGETS`] targets, plans (baseline or PSO), passes the plan through the
//! safety gate, executes it, and broadcasts a claim for each observed target
//! so peers do not waste energy re-observing it within the current
//! [`EPOCH_TICKS`] observation window. Communication faults make
//! duplicated observations possible; the metrics measure how much that costs.
//!
//! Everything is fixed-size and allocation-free. Given the same
//! [`SimConfig`], [`run`] returns identical [`Metrics`] on every platform
//! with IEEE-754 `f32`.

use crate::baseline::{MissionState, Observation};
use crate::envelope::{Envelope, Kind, ReplayGuard, ENVELOPE_LEN};
use crate::planner::{self, Plan, MAX_CANDIDATES};
use crate::rng::SplitMix64;
use crate::safety::{self, Verdict};
use crate::state::{AbStore, Snapshot, SEQ_RESTART_GAP};

pub const N_CRAFT: usize = 3;
pub const N_TARGETS: usize = 64;
pub const QUEUE_CAP: usize = 256;
pub const SNAPSHOT_EVERY: u32 = 10;
pub const MAX_BLACKOUTS: usize = 4;
/// Science is renewable: every epoch is a new observation window in which
/// every target can be observed once and carries a fresh science value.
/// Claims from an earlier epoch are ignored.
pub const EPOCH_TICKS: u32 = 50;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Baseline,
    Pso,
}

/// Half-open tick interval `[start, end)` during which all links are down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub start: u32,
    pub end: u32,
}

/// Deterministic fault schedule. A value of `0` for an `*_every` field means
/// "never"; otherwise every N-th event (1-based global count) is affected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Faults {
    pub blackouts: [Option<Window>; MAX_BLACKOUTS],
    pub drop_every: u32,
    pub duplicate_every: u32,
    pub corrupt_every: u32,
    pub delay_every: u32,
    pub extra_delay: u32,
    pub corrupt_observation_every: u32,
    /// Reboot spacecraft `.0` at tick `.1`, restoring from persistent state.
    pub restart: Option<(usize, u32)>,
    /// Corrupt the newest snapshot just before that restart.
    pub corrupt_snapshot_before_restart: bool,
}

impl Faults {
    pub const NONE: Faults = Faults {
        blackouts: [None; MAX_BLACKOUTS],
        drop_every: 0,
        duplicate_every: 0,
        corrupt_every: 0,
        delay_every: 0,
        extra_delay: 0,
        corrupt_observation_every: 0,
        restart: None,
        corrupt_snapshot_before_restart: false,
    };

    /// Every fault class at once; used by the stress tests and the demo.
    pub const fn harsh() -> Faults {
        Faults {
            blackouts: [
                Some(Window { start: 50, end: 90 }),
                Some(Window {
                    start: 200,
                    end: 260,
                }),
                None,
                None,
            ],
            drop_every: 7,
            duplicate_every: 5,
            corrupt_every: 11,
            delay_every: 4,
            extra_delay: 6,
            corrupt_observation_every: 9,
            restart: Some((1, 137)),
            corrupt_snapshot_before_restart: true,
        }
    }

    fn in_blackout(&self, t: u32) -> bool {
        self.blackouts
            .iter()
            .flatten()
            .any(|w| t >= w.start && t < w.end)
    }
}

fn hits(every: u32, count: u32) -> bool {
    every != 0 && count % every == 0
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimConfig {
    pub seed: u64,
    pub ticks: u32,
    pub strategy: Strategy,
    pub faults: Faults,
    pub capacity: f32,
    pub reserve: f32,
    pub recharge: f32,
    pub base_delay: u32,
    pub planner_iterations: u32,
}

impl SimConfig {
    pub fn nominal(seed: u64, strategy: Strategy) -> Self {
        Self {
            seed,
            ticks: 400,
            strategy,
            faults: Faults::NONE,
            capacity: 40.0,
            reserve: 10.0,
            recharge: 0.5,
            base_delay: 2,
            planner_iterations: 40,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    pub decisions: u32,
    pub observations: u32,
    pub unique_science: f32,
    pub duplicate_observations: u32,
    pub energy_used: f32,
    pub conserve_decisions: u32,
    pub safety_rejections: u32,
    pub fallbacks: u32,
    pub corrupted_candidates: u32,
    pub messages_sent: u32,
    pub messages_dropped: u32,
    pub messages_delivered: u32,
    pub messages_accepted: u32,
    pub rejected_decode: u32,
    pub rejected_replay: u32,
    /// Authentic claims that arrived after their epoch ended.
    pub stale_claims: u32,
    pub queue_overflow: u32,
    pub restarts: u32,
    pub snapshot_fallbacks: u32,
    pub cold_starts: u32,
    /// Lowest `energy - reserve` seen after any action. Must stay `>= 0`.
    pub min_reserve_margin: f32,
    pub max_evals_per_decision: u32,
    pub total_evals: u32,
}

impl Metrics {
    const ZERO: Metrics = Metrics {
        decisions: 0,
        observations: 0,
        unique_science: 0.0,
        duplicate_observations: 0,
        energy_used: 0.0,
        conserve_decisions: 0,
        safety_rejections: 0,
        fallbacks: 0,
        corrupted_candidates: 0,
        messages_sent: 0,
        messages_dropped: 0,
        messages_delivered: 0,
        messages_accepted: 0,
        rejected_decode: 0,
        rejected_replay: 0,
        stale_claims: 0,
        queue_overflow: 0,
        restarts: 0,
        snapshot_fallbacks: 0,
        cold_starts: 0,
        min_reserve_margin: f32::INFINITY,
        max_evals_per_decision: 0,
        total_evals: 0,
    };
}

#[derive(Clone, Copy, Debug)]
struct Craft {
    id: u16,
    energy: f32,
    known: u64,
    next_seq: u32,
    rng: SplitMix64,
    replay: ReplayGuard,
    store: AbStore,
    generation: u32,
}

#[derive(Clone, Copy, Debug)]
struct InFlight {
    deliver_at: u32,
    to: usize,
    bytes: [u8; ENVELOPE_LEN],
}

/// Complete simulation state. Its size is fixed at compile time; see
/// [`state_bytes`].
pub struct Sim {
    cfg: SimConfig,
    crafts: [Craft; N_CRAFT],
    catalog: [f32; N_TARGETS],
    catalog_rng: SplitMix64,
    truth_observed: u64,
    queue: [Option<InFlight>; QUEUE_CAP],
    msg_count: u32,
    cand_count: u32,
    m: Metrics,
}

/// Bytes of simulation state (no heap is used anywhere).
pub const fn state_bytes() -> usize {
    core::mem::size_of::<Sim>()
}

fn craft_id(i: usize) -> u16 {
    (i as u16) + 1
}

impl Sim {
    pub fn new(cfg: SimConfig) -> Self {
        let mut root = SplitMix64::new(cfg.seed);
        let mut catalog_rng = root.fork(0x0CA7_A106);
        let mut catalog = [0.0f32; N_TARGETS];
        for v in catalog.iter_mut() {
            *v = catalog_rng.range(1.0, 10.0);
        }
        let mut crafts = [Craft {
            id: 0,
            energy: 0.0,
            known: 0,
            next_seq: 0,
            rng: SplitMix64::new(0),
            replay: ReplayGuard::new(&[]),
            store: AbStore::new(),
            generation: 0,
        }; N_CRAFT];
        for (i, c) in crafts.iter_mut().enumerate() {
            let mut peers = [0u16; N_CRAFT - 1];
            let mut k = 0;
            for j in 0..N_CRAFT {
                if j != i {
                    peers[k] = craft_id(j);
                    k += 1;
                }
            }
            c.id = craft_id(i);
            c.energy = cfg.capacity;
            c.rng = root.fork(i as u64);
            c.replay = ReplayGuard::new(&peers);
        }
        Sim {
            cfg,
            crafts,
            catalog,
            catalog_rng,
            truth_observed: 0,
            queue: [None; QUEUE_CAP],
            msg_count: 0,
            cand_count: 0,
            m: Metrics::ZERO,
        }
    }

    pub fn run(mut self) -> Metrics {
        for t in 0..self.cfg.ticks {
            self.step(t);
        }
        // Drain anything still in flight so message accounting is exact.
        // Late arrivals land after the final epoch and count as stale.
        self.deliver(u32::MAX);
        self.m
    }

    fn step(&mut self, t: u32) {
        if t > 0 && t % EPOCH_TICKS == 0 {
            self.new_epoch();
        }
        if let Some((idx, at)) = self.cfg.faults.restart {
            if at == t && idx < N_CRAFT {
                self.restart(idx, t);
            }
        }
        self.deliver(t);
        for i in 0..N_CRAFT {
            self.act(i, t);
        }
    }

    fn new_epoch(&mut self) {
        self.truth_observed = 0;
        for v in self.catalog.iter_mut() {
            *v = self.catalog_rng.range(1.0, 10.0);
        }
        for c in self.crafts.iter_mut() {
            c.known = 0;
        }
    }

    fn restart(&mut self, i: usize, t: u32) {
        self.m.restarts += 1;
        let c = &mut self.crafts[i];
        if self.cfg.faults.corrupt_snapshot_before_restart {
            c.store.corrupt_latest(13);
        }
        match c.store.restore() {
            Some(s) => {
                if s.generation != c.generation {
                    self.m.snapshot_fallbacks += 1;
                }
                // Energy is physical (battery charge), not software state,
                // so it survives the reboot. Everything else comes from NVM.
                c.known = if s.tick / EPOCH_TICKS == t / EPOCH_TICKS {
                    s.known_observed
                } else {
                    0
                };
                c.rng = SplitMix64::new(s.rng_state);
                c.replay = s.replay;
                c.next_seq = s.next_seq.wrapping_add(SEQ_RESTART_GAP);
                c.generation = s.generation;
            }
            None => {
                self.m.cold_starts += 1;
                c.known = 0;
            }
        }
    }

    fn deliver(&mut self, t: u32) {
        for slot in 0..QUEUE_CAP {
            let msg = match self.queue[slot] {
                Some(m) if m.deliver_at <= t => m,
                _ => continue,
            };
            self.queue[slot] = None;
            self.m.messages_delivered += 1;
            let env = match Envelope::decode(&msg.bytes) {
                Ok(e) => e,
                Err(_) => {
                    self.m.rejected_decode += 1;
                    continue;
                }
            };
            let c = &mut self.crafts[msg.to];
            if c.replay.accept(env.sender, env.seq).is_err() {
                self.m.rejected_replay += 1;
                continue;
            }
            self.m.messages_accepted += 1;
            if env.tick / EPOCH_TICKS != t / EPOCH_TICKS {
                self.m.stale_claims += 1;
                continue;
            }
            if env.kind == Kind::Claim && (env.target as usize) < N_TARGETS {
                c.known |= 1u64 << env.target;
            }
        }
    }

    fn candidates(&mut self, i: usize, out: &mut [Observation; MAX_CANDIDATES]) -> usize {
        let mut n = 0;
        // Bounded number of draws; fewer candidates appear as the catalog
        // is exhausted.
        for _ in 0..(2 * MAX_CANDIDATES) {
            if n == MAX_CANDIDATES {
                break;
            }
            let c = &mut self.crafts[i];
            let target = c.rng.below(N_TARGETS as u32);
            let cost = c.rng.range(1.0, 6.0);
            if c.known & (1u64 << target) != 0 || out[..n].iter().any(|o| o.id == target) {
                continue;
            }
            self.cand_count += 1;
            let mut value = self.catalog[target as usize];
            if hits(self.cfg.faults.corrupt_observation_every, self.cand_count) {
                self.m.corrupted_candidates += 1;
                value = match self.cand_count % 3 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    _ => -value,
                };
            }
            out[n] = Observation {
                id: target,
                science_value: value,
                energy_cost: cost,
            };
            n += 1;
        }
        n
    }

    fn act(&mut self, i: usize, t: u32) {
        {
            let c = &mut self.crafts[i];
            c.energy = (c.energy + self.cfg.recharge).min(self.cfg.capacity);
        }
        let mut buf = [Observation::EMPTY; MAX_CANDIDATES];
        let n = self.candidates(i, &mut buf);
        let cands = &buf[..n];
        let state = MissionState {
            available_energy: self.crafts[i].energy,
            reserve_energy: self.cfg.reserve,
        };

        self.m.decisions += 1;
        let (plan, evals) = match self.cfg.strategy {
            Strategy::Baseline => (Plan::from_decision(state.choose(cands)), 0),
            Strategy::Pso => {
                let seed = self.crafts[i].rng.next_u64();
                let pcfg = planner::planner_config(self.cfg.planner_iterations);
                planner::propose(cands, &state, &pcfg, seed).unwrap_or((Plan::EMPTY, 0))
            }
        };
        self.m.total_evals += evals;
        self.m.max_evals_per_decision = self.m.max_evals_per_decision.max(evals);

        let (plan, energy) = match safety::review(&plan, cands, &state) {
            Verdict::Approved { energy } => (plan, energy),
            Verdict::Rejected(_) => {
                self.m.safety_rejections += 1;
                self.m.fallbacks += 1;
                let fb = Plan::from_decision(state.choose(cands));
                match safety::review(&fb, cands, &state) {
                    Verdict::Approved { energy } => (fb, energy),
                    Verdict::Rejected(_) => (Plan::EMPTY, 0.0),
                }
            }
        };

        if plan.len == 0 {
            self.m.conserve_decisions += 1;
        }
        self.crafts[i].energy -= energy;
        self.m.energy_used += energy;
        let margin = self.crafts[i].energy - self.cfg.reserve;
        if margin < self.m.min_reserve_margin {
            self.m.min_reserve_margin = margin;
        }

        for &id in plan.ids() {
            let bit = 1u64 << id;
            self.m.observations += 1;
            if self.truth_observed & bit != 0 {
                self.m.duplicate_observations += 1;
            } else {
                self.m.unique_science += self.catalog[id as usize];
                self.truth_observed |= bit;
            }
            self.crafts[i].known |= bit;
            let value = cands
                .iter()
                .find(|o| o.id == id)
                .map_or(0.0, |o| o.science_value);
            for j in 0..N_CRAFT {
                if j != i {
                    self.send(i, j, id, value, t);
                }
            }
        }

        if t % SNAPSHOT_EVERY == 0 {
            let c = &mut self.crafts[i];
            c.generation = c.generation.wrapping_add(1);
            let s = Snapshot {
                generation: c.generation,
                spacecraft_id: c.id,
                tick: t,
                known_observed: c.known,
                rng_state: c.rng.state(),
                next_seq: c.next_seq,
                energy: c.energy,
                replay: c.replay,
            };
            c.store.write(&s);
        }
    }

    fn send(&mut self, from: usize, to: usize, target: u32, value: f32, t: u32) {
        let c = &mut self.crafts[from];
        let env = Envelope {
            kind: Kind::Claim,
            sender: c.id,
            seq: c.next_seq,
            tick: t,
            target,
            value,
        };
        c.next_seq = c.next_seq.wrapping_add(1);
        self.msg_count += 1;
        self.m.messages_sent += 1;
        let f = self.cfg.faults;
        if f.in_blackout(t) || hits(f.drop_every, self.msg_count) {
            self.m.messages_dropped += 1;
            return;
        }
        let mut bytes = env.encode();
        if hits(f.corrupt_every, self.msg_count) {
            bytes[(self.msg_count as usize) % ENVELOPE_LEN] ^= 0x40;
        }
        let mut delay = self.cfg.base_delay;
        if hits(f.delay_every, self.msg_count) {
            delay += f.extra_delay;
        }
        let msg = InFlight {
            deliver_at: t + 1 + delay,
            to,
            bytes,
        };
        self.enqueue(msg);
        if hits(f.duplicate_every, self.msg_count) {
            self.enqueue(InFlight {
                deliver_at: msg.deliver_at + 1,
                ..msg
            });
        }
    }

    fn enqueue(&mut self, msg: InFlight) {
        match self.queue.iter_mut().find(|s| s.is_none()) {
            Some(slot) => *slot = Some(msg),
            None => self.m.queue_overflow += 1,
        }
    }
}

pub fn run(cfg: &SimConfig) -> Metrics {
    Sim::new(*cfg).run()
}
