// Copyright (c) 2026 pykep-lambert contributors
// SPDX-License-Identifier: MPL-2.0

//! Fixed KTTSP Lambert-leg objective using the published Rust crates.

use fcmaes_core::{NAN_REPLACEMENT, Objective};
use pykep_core::astro::elements::ClassicalElements;
use pykep_core::astro::lambert::LambertProblem;
use pykep_core::ephemeris::{Ephemeris, KeplerianEphemeris};
use pykep_core::time::epoch::Epoch;

/// Exact `pykep-lambert` revision used to define this workload.
pub const SOURCE_COMMIT: &str = "e1e4bb36a9e26470e0f8268180cd3c3c77a48443";

/// Seconds in one benchmark day.
pub const DAY_SECONDS: f64 = 86_400.0;

/// Lunar gravitational parameter used by the KTTSP instance, in m³/s².
pub const MU_MOON: f64 = 4.904_869_5e12;

/// Minimum allowed time of flight, in days.
pub const MIN_TOF_DAYS: f64 = 0.001;

/// Maximum allowed wait, in days.
pub const MAX_WAIT_DAYS: f64 = 12.0;

/// Maximum allowed time of flight, in days.
pub const MAX_TOF_DAYS: f64 = 12.0;

/// Delta-v limit above which the objective applies a penalty, in m/s.
pub const DV_LIMIT: f64 = 600.0;

/// Delta-v excess accepted as feasible when reporting optimizer results.
pub const DV_TOLERANCE: f64 = 1.0e-7;

/// Linear delta-v excess penalty, in days per m/s.
pub const PENALTY_LINEAR: f64 = 1_000.0;

/// Quadratic delta-v excess penalty, in days per (m/s)².
pub const PENALTY_QUADRATIC: f64 = 1.0;

/// Maximum number of complete revolutions searched for the fixed outer leg.
pub const MAXIMUM_REVOLUTIONS: usize = 2;

const ORBITAL_ROWS: [[f64; 6]; 2] = [
    [
        1.496_861_100e7,
        7.282_163_486e-3,
        1.053_864_218,
        3.677_831_327e-3,
        3.972_894_704,
        3.980_584_570,
    ],
    [
        1.497_827_726e7,
        1.158_690_595e-3,
        1.593_610_788,
        6.232_981_268e-3,
        2.079_093_608,
        3.993_488_927e-1,
    ],
];

/// Wait-time and time-of-flight decision, both in days.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    /// Delay from the current epoch to departure.
    pub wait_days: f64,
    /// Lambert time of flight.
    pub tof_days: f64,
}

impl Decision {
    /// Constructs a fixed-leg decision in days.
    #[must_use]
    pub const fn new(wait_days: f64, tof_days: f64) -> Self {
        Self {
            wait_days,
            tof_days,
        }
    }
}

/// Complete result of one valid fixed-leg objective evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evaluation {
    /// Input decision.
    pub decision: Decision,
    /// Departure epoch relative to MJD2000, in days.
    pub departure_day: f64,
    /// Arrival epoch relative to MJD2000, in days.
    pub arrival_day: f64,
    /// Best two-impulse delta-v across all searched Lambert branches, in m/s.
    pub delta_v: f64,
    /// Delta-v above [`DV_LIMIT`], in m/s.
    pub excess_delta_v: f64,
    /// Penalized scalar objective in days.
    pub objective: f64,
}

/// Self-contained fixed leg from body zero to body one of `easy.kttsp`.
#[derive(Clone, Debug)]
pub struct FixedLegProblem {
    bodies: [KeplerianEphemeris; 2],
}

impl FixedLegProblem {
    /// Builds the benchmark problem at MJD2000 zero.
    ///
    /// # Errors
    ///
    /// Returns a `pykep-core` validation or conversion error if the embedded
    /// orbital elements cannot form a Keplerian ephemeris.
    pub fn easy() -> pykep_core::Result<Self> {
        let epoch = Epoch::from_mjd2000(0.0)?;
        let make_body = |index: usize| {
            KeplerianEphemeris::from_classical_mean(
                epoch,
                ClassicalElements::from(ORBITAL_ROWS[index]),
                MU_MOON,
                format!("easy.kttsp body {index}"),
                None,
                None,
                None,
            )
        };
        Ok(Self {
            bodies: [make_body(0)?, make_body(1)?],
        })
    }

    /// Evaluates a decision and returns all physical and objective values.
    ///
    /// `None` denotes a non-finite or out-of-bounds decision, an ephemeris
    /// failure, or a geometry for which neither Lambert direction succeeds.
    #[must_use]
    pub fn evaluate(&self, decision: Decision) -> Option<Evaluation> {
        if !decision.wait_days.is_finite()
            || !decision.tof_days.is_finite()
            || !(0.0..=MAX_WAIT_DAYS).contains(&decision.wait_days)
            || !(MIN_TOF_DAYS..=MAX_TOF_DAYS).contains(&decision.tof_days)
        {
            return None;
        }

        let departure_day = decision.wait_days;
        let arrival_day = departure_day + decision.tof_days;
        let departure = self.bodies[0].state(departure_day).ok()?;
        let arrival = self.bodies[1].state(arrival_day).ok()?;
        let initial_position = [departure[0], departure[1], departure[2]];
        let final_position = [arrival[0], arrival[1], arrival[2]];
        let initial_velocity = [departure[3], departure[4], departure[5]];
        let final_velocity = [arrival[3], arrival[4], arrival[5]];
        let mut delta_v = f64::INFINITY;

        for clockwise in [false, true] {
            let Ok(lambert) = LambertProblem::new(
                initial_position,
                final_position,
                decision.tof_days * DAY_SECONDS,
                MU_MOON,
                clockwise,
                MAXIMUM_REVOLUTIONS,
            ) else {
                continue;
            };
            for solution in lambert.solutions() {
                let departure_impulse =
                    difference_norm(solution.departure_velocity, initial_velocity);
                let arrival_impulse = difference_norm(solution.arrival_velocity, final_velocity);
                delta_v = delta_v.min(departure_impulse + arrival_impulse);
            }
        }
        if !delta_v.is_finite() {
            return None;
        }

        let excess_delta_v = (delta_v - DV_LIMIT).max(0.0);
        let objective = arrival_day
            + PENALTY_LINEAR * excess_delta_v
            + PENALTY_QUADRATIC * excess_delta_v * excess_delta_v;
        Some(Evaluation {
            decision,
            departure_day,
            arrival_day,
            delta_v,
            excess_delta_v,
            objective,
        })
    }

    /// Returns the scalar objective or the optimizer's finite replacement.
    #[must_use]
    pub fn objective(&self, decision: Decision) -> f64 {
        self.evaluate(decision)
            .map_or(NAN_REPLACEMENT, |evaluation| evaluation.objective)
    }
}

impl Objective for FixedLegProblem {
    fn nobj(&self) -> usize {
        1
    }

    fn eval(&self, x: &[f64]) -> Vec<f64> {
        vec![self.eval_scalar(x)]
    }

    fn eval_scalar(&self, x: &[f64]) -> f64 {
        if x.len() != 2 {
            return NAN_REPLACEMENT;
        }
        self.objective(Decision::new(x[0], x[1]))
    }
}

fn difference_norm(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE_RELATIVE_TOLERANCE: f64 = 2.0e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= REFERENCE_RELATIVE_TOLERANCE * scale,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn matches_pykep_reference_without_penalty() {
        let problem = FixedLegProblem::easy().unwrap();
        let evaluation = problem.evaluate(Decision::new(0.0, 1.0)).unwrap();
        assert_close(evaluation.delta_v, 539.922_343_703_232_4);
        assert_eq!(evaluation.excess_delta_v, 0.0);
        assert_eq!(evaluation.objective, 1.0);
    }

    #[test]
    fn matches_numba_reference_with_penalty() {
        let problem = FixedLegProblem::easy().unwrap();
        let evaluation = problem.evaluate(Decision::new(12.0, 12.0)).unwrap();
        assert_close(evaluation.delta_v, 828.468_729_029_679_3);
        assert_close(evaluation.objective, 280_690.689_174_113_9);
    }

    #[test]
    fn rejects_wrong_shape_and_invalid_bounds() {
        let problem = FixedLegProblem::easy().unwrap();
        assert_eq!(problem.eval_scalar(&[1.0]), NAN_REPLACEMENT);
        assert!(problem.evaluate(Decision::new(-0.1, 1.0)).is_none());
        assert!(problem.evaluate(Decision::new(0.0, 0.0)).is_none());
        assert!(problem.evaluate(Decision::new(0.0, f64::NAN)).is_none());
    }
}
