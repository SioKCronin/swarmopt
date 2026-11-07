//! Safety gate: the only path from a proposed plan to an executed action.
//!
//! The gate re-derives every quantity it checks from the candidate list and
//! the mission state; it never trusts totals reported by the optimizer.

use crate::baseline::{MissionState, Observation};
use crate::planner::{Plan, MAX_CANDIDATES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Energy state is non-finite, negative reserve, or already below reserve.
    UntrustedEnergyState,
    /// Plan length is out of range.
    TooManyActions,
    /// Plan references an id not in this tick's candidate list.
    UnknownObservation,
    /// Plan references the same observation twice.
    DuplicateObservation,
    /// Plan references a candidate with non-finite or out-of-range fields.
    InvalidObservation,
    /// Executing the plan would drop energy below reserve.
    ReserveViolated,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// Approved, with the energy the plan will consume.
    Approved {
        energy: f32,
    },
    Rejected(RejectReason),
}

pub fn review(plan: &Plan, candidates: &[Observation], state: &MissionState) -> Verdict {
    let budget = match state.spendable() {
        Some(b) => b,
        None => {
            // An empty plan (conserve) is always safe.
            return if plan.len == 0 {
                Verdict::Approved { energy: 0.0 }
            } else {
                Verdict::Rejected(RejectReason::UntrustedEnergyState)
            };
        }
    };
    if plan.len > MAX_CANDIDATES {
        return Verdict::Rejected(RejectReason::TooManyActions);
    }
    let ids = plan.ids();
    let mut energy = 0.0f32;
    for (k, id) in ids.iter().enumerate() {
        if ids[..k].contains(id) {
            return Verdict::Rejected(RejectReason::DuplicateObservation);
        }
        let obs = match candidates.iter().find(|o| o.id == *id) {
            Some(o) => o,
            None => return Verdict::Rejected(RejectReason::UnknownObservation),
        };
        if !obs.is_valid() {
            return Verdict::Rejected(RejectReason::InvalidObservation);
        }
        energy += obs.energy_cost;
    }
    if !energy.is_finite() || energy > budget {
        return Verdict::Rejected(RejectReason::ReserveViolated);
    }
    Verdict::Approved { energy }
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

    fn plan(ids: &[u32]) -> Plan {
        let mut p = Plan::EMPTY;
        p.ids[..ids.len()].copy_from_slice(ids);
        p.len = ids.len();
        p
    }

    const S: MissionState = MissionState {
        available_energy: 10.0,
        reserve_energy: 4.0,
    };

    #[test]
    fn approves_within_budget() {
        let c = [obs(1, 1.0, 3.0), obs(2, 1.0, 3.0)];
        assert_eq!(
            review(&plan(&[1, 2]), &c, &S),
            Verdict::Approved { energy: 6.0 }
        );
    }

    #[test]
    fn rejects_reserve_violation() {
        let c = [obs(1, 1.0, 3.0), obs(2, 1.0, 3.5)];
        assert_eq!(
            review(&plan(&[1, 2]), &c, &S),
            Verdict::Rejected(RejectReason::ReserveViolated)
        );
    }

    #[test]
    fn rejects_unknown_duplicate_invalid() {
        let c = [obs(1, 1.0, 1.0), obs(2, f32::NAN, 1.0)];
        assert_eq!(
            review(&plan(&[9]), &c, &S),
            Verdict::Rejected(RejectReason::UnknownObservation)
        );
        assert_eq!(
            review(&plan(&[1, 1]), &c, &S),
            Verdict::Rejected(RejectReason::DuplicateObservation)
        );
        assert_eq!(
            review(&plan(&[2]), &c, &S),
            Verdict::Rejected(RejectReason::InvalidObservation)
        );
    }

    #[test]
    fn conserve_is_always_safe() {
        let bad = MissionState {
            available_energy: f32::NAN,
            reserve_energy: 1.0,
        };
        assert_eq!(
            review(&Plan::EMPTY, &[], &bad),
            Verdict::Approved { energy: 0.0 }
        );
        assert_eq!(
            review(&plan(&[1]), &[obs(1, 1.0, 1.0)], &bad),
            Verdict::Rejected(RejectReason::UntrustedEnergyState)
        );
    }

    #[test]
    fn rejects_oversized_plan_len() {
        let mut p = Plan::EMPTY;
        p.len = MAX_CANDIDATES + 1;
        assert_eq!(
            review(&p, &[], &S),
            Verdict::Rejected(RejectReason::TooManyActions)
        );
    }
}
