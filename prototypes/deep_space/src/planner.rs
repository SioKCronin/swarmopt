//! PSO-based observation planner.
//!
//! Encodes "which of up to [`MAX_CANDIDATES`] observations to take this tick"
//! as a point in `[0, 1]^MAX_CANDIDATES` (a coordinate `>= 0.5` selects that
//! candidate) and lets bounded PSO maximize total science value with a steep
//! penalty for exceeding the spendable energy budget.
//!
//! The penalty only *steers* the optimizer. The planner does not guarantee a
//! feasible plan — that is the [`crate::safety`] gate's job.

use crate::baseline::{Decision, MissionState, Observation};
use crate::pso::{self, PsoConfig, PsoError};

pub const MAX_CANDIDATES: usize = 8;
pub const PLANNER_PARTICLES: usize = 16;
const SELECT_THRESHOLD: f32 = 0.5;
const OVERBUDGET_PENALTY: f32 = 1_000.0;

/// A set of observations to perform in one tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub ids: [u32; MAX_CANDIDATES],
    pub len: usize,
}

impl Plan {
    pub const EMPTY: Plan = Plan {
        ids: [0; MAX_CANDIDATES],
        len: 0,
    };

    pub fn ids(&self) -> &[u32] {
        &self.ids[..self.len.min(MAX_CANDIDATES)]
    }

    pub fn single(id: u32) -> Plan {
        let mut p = Plan::EMPTY;
        p.ids[0] = id;
        p.len = 1;
        p
    }

    pub fn from_decision(d: Decision) -> Plan {
        match d {
            Decision::Observe(id) => Plan::single(id),
            Decision::ConserveEnergy => Plan::EMPTY,
        }
    }
}

pub fn planner_config(iterations: u32) -> PsoConfig {
    let mut c = PsoConfig::standard(0.0, 1.0, iterations);
    c.v_max = 0.5;
    c
}

/// Propose a plan. `candidates` may contain corrupted entries; they are scored
/// as zero-value so the optimizer has no incentive to pick them, but they are
/// *not* filtered here — the safety gate rejects any plan that includes them.
pub fn propose(
    candidates: &[Observation],
    state: &MissionState,
    cfg: &PsoConfig,
    seed: u64,
) -> Result<(Plan, u32), PsoError> {
    let n = candidates.len().min(MAX_CANDIDATES);
    let budget = state.spendable().unwrap_or(0.0);

    let objective = |x: &[f32; MAX_CANDIDATES]| -> f32 {
        let mut value = 0.0;
        let mut energy = 0.0;
        for (xi, o) in x.iter().zip(candidates.iter().take(n)) {
            if *xi >= SELECT_THRESHOLD {
                if o.is_valid() {
                    value += o.science_value;
                    energy += o.energy_cost;
                } else {
                    energy += OVERBUDGET_PENALTY;
                }
            }
        }
        let over = if energy > budget {
            energy - budget
        } else {
            0.0
        };
        -value + OVERBUDGET_PENALTY * over
    };

    let r = pso::minimize::<PLANNER_PARTICLES, MAX_CANDIDATES, _>(cfg, seed, objective)?;

    let mut plan = Plan::EMPTY;
    for (xi, o) in r.best_position.iter().zip(candidates.iter().take(n)) {
        if *xi >= SELECT_THRESHOLD {
            plan.ids[plan.len] = o.id;
            plan.len += 1;
        }
    }
    Ok((plan, r.evaluations))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(id: u32, v: f32, c: f32) -> Observation {
        Observation {
            id,
            science_value: v,
            energy_cost: c,
        }
    }

    #[test]
    fn finds_knapsack_optimum_on_small_instance() {
        // Budget 10. Best subset: {2, 3} -> value 14, energy 10.
        // Baseline (best ratio, single pick) would choose id 1 (ratio 5).
        let c = [
            obs(1, 5.0, 1.0),
            obs(2, 8.0, 6.0),
            obs(3, 6.0, 4.0),
            obs(4, 9.0, 9.0),
        ];
        let s = MissionState {
            available_energy: 12.0,
            reserve_energy: 2.0,
        };
        let (plan, evals) = propose(&c, &s, &planner_config(60), 1).unwrap();
        let mut ids = [0u32; MAX_CANDIDATES];
        ids[..plan.len].copy_from_slice(plan.ids());
        ids[..plan.len].sort_unstable();
        // {1,2,3} costs 11 > 10; {1,2} = 13 value/7 energy; {2,3} = 14/10;
        // {1,3} = 11/5; {1,4} = 14/10. Optimum value is 14.
        let value: f32 = plan
            .ids()
            .iter()
            .map(|id| c.iter().find(|o| o.id == *id).unwrap().science_value)
            .sum();
        assert_eq!(value, 14.0, "plan = {:?}", plan.ids());
        assert_eq!(evals, (PLANNER_PARTICLES as u32) * 61);
    }

    #[test]
    fn empty_when_nothing_affordable() {
        let c = [obs(1, 5.0, 50.0)];
        let s = MissionState {
            available_energy: 10.0,
            reserve_energy: 5.0,
        };
        let (plan, _) = propose(&c, &s, &planner_config(30), 2).unwrap();
        assert_eq!(plan.len, 0);
    }
}
