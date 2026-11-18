//! Deterministic observation-prioritization baseline (not PSO).
//!
//! Picks the single observation with the best science-value-per-energy ratio
//! that fits inside the energy budget above reserve. Every input is validated;
//! anything non-finite or out of range is skipped, and any doubt about the
//! energy state results in [`Decision::ConserveEnergy`].

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    pub id: u32,
    pub science_value: f32,
    pub energy_cost: f32,
}

impl Observation {
    pub const EMPTY: Observation = Observation {
        id: u32::MAX,
        science_value: 0.0,
        energy_cost: 0.0,
    };

    /// Structural validity: finite, non-negative value, strictly positive cost.
    pub fn is_valid(&self) -> bool {
        self.science_value.is_finite()
            && self.energy_cost.is_finite()
            && self.science_value >= 0.0
            && self.energy_cost > 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    Observe(u32),
    ConserveEnergy,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MissionState {
    pub available_energy: f32,
    pub reserve_energy: f32,
}

impl MissionState {
    /// Energy that may be spent this decision without touching reserve, or
    /// `None` if the energy state itself is untrustworthy.
    pub fn spendable(&self) -> Option<f32> {
        if !self.available_energy.is_finite()
            || !self.reserve_energy.is_finite()
            || self.reserve_energy < 0.0
            || self.available_energy < self.reserve_energy
        {
            return None;
        }
        Some(self.available_energy - self.reserve_energy)
    }

    pub fn choose(&self, observations: &[Observation]) -> Decision {
        let budget = match self.spendable() {
            Some(b) => b,
            None => return Decision::ConserveEnergy,
        };
        let mut best = None;
        let mut best_score = f32::NEG_INFINITY;
        for obs in observations {
            if !obs.is_valid() || obs.energy_cost > budget {
                continue;
            }
            let score = obs.science_value / obs.energy_cost;
            // Strict `>` keeps ties deterministic: first candidate wins.
            if score > best_score {
                best = Some(obs.id);
                best_score = score;
            }
        }
        match best {
            Some(id) => Decision::Observe(id),
            None => Decision::ConserveEnergy,
        }
    }
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
    fn picks_best_ratio_within_budget() {
        let s = MissionState {
            available_energy: 10.0,
            reserve_energy: 4.0,
        };
        let o = [obs(1, 10.0, 5.0), obs(2, 9.0, 3.0), obs(3, 100.0, 7.0)];
        // id 3 exceeds the 6.0 budget; id 2 has ratio 3.0 > 2.0.
        assert_eq!(s.choose(&o), Decision::Observe(2));
    }

    #[test]
    fn conserves_below_reserve_or_on_bad_state() {
        let o = [obs(1, 1.0, 1.0)];
        assert_eq!(
            MissionState {
                available_energy: 3.0,
                reserve_energy: 4.0
            }
            .choose(&o),
            Decision::ConserveEnergy
        );
        assert_eq!(
            MissionState {
                available_energy: f32::NAN,
                reserve_energy: 1.0
            }
            .choose(&o),
            Decision::ConserveEnergy
        );
        assert_eq!(
            MissionState {
                available_energy: 5.0,
                reserve_energy: -1.0
            }
            .choose(&o),
            Decision::ConserveEnergy
        );
    }

    #[test]
    fn skips_corrupted_observations() {
        let s = MissionState {
            available_energy: 10.0,
            reserve_energy: 0.0,
        };
        let o = [
            obs(1, f32::NAN, 1.0),
            obs(2, f32::INFINITY, 1.0),
            obs(3, 5.0, 0.0),
            obs(4, -5.0, 1.0),
            obs(5, 1.0, 2.0),
        ];
        assert_eq!(s.choose(&o), Decision::Observe(5));
    }
}
