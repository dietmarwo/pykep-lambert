// Copyright (c) 2026 pykep-lambert contributors
// SPDX-License-Identifier: MPL-2.0

//! Fixed KTTSP Lambert-leg objective using the published Rust crates.

use fcmaes_core::{NAN_REPLACEMENT, Objective, parallel_batch};
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

/// One ephemeris propagation request for a body in [`FixedLegProblem`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropagationRequest {
    /// Zero-based body index.
    pub body_index: usize,
    /// Requested epoch relative to MJD2000, in days.
    pub epoch_day: f64,
}

impl PropagationRequest {
    /// Constructs a propagation request.
    #[must_use]
    pub const fn new(body_index: usize, epoch_day: f64) -> Self {
        Self {
            body_index,
            epoch_day,
        }
    }
}

/// Pre-propagated states and parameters for one two-impulse Lambert transfer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LambertTransfer {
    /// Cartesian departure state `[x, y, z, vx, vy, vz]` in SI units.
    pub departure_state: [f64; 6],
    /// Cartesian arrival state `[x, y, z, vx, vy, vz]` in SI units.
    pub arrival_state: [f64; 6],
    /// Time of flight in seconds.
    pub tof_seconds: f64,
    /// Central-body gravitational parameter in m³/s².
    pub mu: f64,
    /// Maximum number of complete revolutions to search.
    pub maximum_revolutions: usize,
}

impl LambertTransfer {
    /// Constructs a pre-propagated Lambert transfer.
    #[must_use]
    pub const fn new(
        departure_state: [f64; 6],
        arrival_state: [f64; 6],
        tof_seconds: f64,
        mu: f64,
        maximum_revolutions: usize,
    ) -> Self {
        Self {
            departure_state,
            arrival_state,
            tof_seconds,
            mu,
            maximum_revolutions,
        }
    }
}

/// Returns the best two-impulse delta-v over both Lambert directions.
///
/// `None` denotes invalid inputs or a geometry for which neither direction
/// produces a finite solution.
#[must_use]
pub fn lambert_delta_v(transfer: &LambertTransfer) -> Option<f64> {
    if !transfer
        .departure_state
        .iter()
        .chain(&transfer.arrival_state)
        .all(|value| value.is_finite())
        || !transfer.tof_seconds.is_finite()
        || transfer.tof_seconds <= 0.0
        || !transfer.mu.is_finite()
        || transfer.mu <= 0.0
    {
        return None;
    }

    let initial_position = [
        transfer.departure_state[0],
        transfer.departure_state[1],
        transfer.departure_state[2],
    ];
    let final_position = [
        transfer.arrival_state[0],
        transfer.arrival_state[1],
        transfer.arrival_state[2],
    ];
    let initial_velocity = [
        transfer.departure_state[3],
        transfer.departure_state[4],
        transfer.departure_state[5],
    ];
    let final_velocity = [
        transfer.arrival_state[3],
        transfer.arrival_state[4],
        transfer.arrival_state[5],
    ];
    let mut delta_v = f64::INFINITY;

    for clockwise in [false, true] {
        let Ok(lambert) = LambertProblem::new(
            initial_position,
            final_position,
            transfer.tof_seconds,
            transfer.mu,
            clockwise,
            transfer.maximum_revolutions,
        ) else {
            continue;
        };
        for solution in lambert.solutions() {
            let departure_impulse = difference_norm(solution.departure_velocity, initial_velocity);
            let arrival_impulse = difference_norm(solution.arrival_velocity, final_velocity);
            delta_v = delta_v.min(departure_impulse + arrival_impulse);
        }
    }
    delta_v.is_finite().then_some(delta_v)
}

/// Evaluates an ordered Lambert batch.
///
/// `workers = 1` is serial, values above one use exactly that many cached
/// Rayon threads, and values at or below zero use the global Rayon pool.
/// Output order always matches input order.
#[must_use]
pub fn lambert_delta_v_batch(transfers: &[LambertTransfer], workers: i32) -> Vec<Option<f64>> {
    parallel_batch(transfers, workers, lambert_delta_v)
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

    /// Propagates one embedded body.
    ///
    /// `None` denotes an invalid body index, non-finite epoch, or ephemeris
    /// failure.
    #[must_use]
    pub fn propagate(&self, request: PropagationRequest) -> Option<[f64; 6]> {
        if !request.epoch_day.is_finite() {
            return None;
        }
        self.bodies
            .get(request.body_index)?
            .state(request.epoch_day)
            .ok()
    }

    /// Propagates an ordered batch of body/epoch requests.
    ///
    /// `workers = 1` is serial, values above one use exactly that many cached
    /// Rayon threads, and values at or below zero use the global Rayon pool.
    /// Output order always matches input order.
    #[must_use]
    pub fn propagate_batch(
        &self,
        requests: &[PropagationRequest],
        workers: i32,
    ) -> Vec<Option<[f64; 6]>> {
        parallel_batch(requests, workers, |request| self.propagate(*request))
    }

    /// Propagates both endpoints of one valid fixed-leg decision.
    #[must_use]
    pub fn transfer(&self, decision: Decision) -> Option<LambertTransfer> {
        if !valid_decision(decision) {
            return None;
        }
        let departure_day = decision.wait_days;
        let arrival_day = departure_day + decision.tof_days;
        let departure_state = self.propagate(PropagationRequest::new(0, departure_day))?;
        let arrival_state = self.propagate(PropagationRequest::new(1, arrival_day))?;
        Some(LambertTransfer::new(
            departure_state,
            arrival_state,
            decision.tof_days * DAY_SECONDS,
            MU_MOON,
            MAXIMUM_REVOLUTIONS,
        ))
    }

    /// Evaluates a decision and returns all physical and objective values.
    ///
    /// `None` denotes a non-finite or out-of-bounds decision, an ephemeris
    /// failure, or a geometry for which neither Lambert direction succeeds.
    #[must_use]
    pub fn evaluate(&self, decision: Decision) -> Option<Evaluation> {
        let transfer = self.transfer(decision)?;
        let delta_v = lambert_delta_v(&transfer)?;
        Some(evaluation_from_delta_v(decision, delta_v))
    }

    /// Evaluates a fixed-leg batch with staged parallelism.
    ///
    /// All departure and arrival ephemerides are propagated in one ordered
    /// batch, followed by one ordered batch containing every valid Lambert
    /// computation. Invalid inputs remain `None` at their original positions.
    #[must_use]
    pub fn evaluate_batch(&self, decisions: &[Decision], workers: i32) -> Vec<Option<Evaluation>> {
        let propagation_requests = decisions
            .iter()
            .flat_map(|decision| {
                if valid_decision(*decision) {
                    [
                        PropagationRequest::new(0, decision.wait_days),
                        PropagationRequest::new(1, decision.wait_days + decision.tof_days),
                    ]
                } else {
                    [
                        PropagationRequest::new(0, f64::NAN),
                        PropagationRequest::new(1, f64::NAN),
                    ]
                }
            })
            .collect::<Vec<_>>();
        let states = self.propagate_batch(&propagation_requests, workers);
        let transfers = decisions
            .iter()
            .enumerate()
            .map(|(index, decision)| {
                Some(LambertTransfer::new(
                    states[index * 2]?,
                    states[index * 2 + 1]?,
                    decision.tof_days * DAY_SECONDS,
                    MU_MOON,
                    MAXIMUM_REVOLUTIONS,
                ))
            })
            .collect::<Vec<_>>();
        let delta_vs = parallel_batch(&transfers, workers, |transfer| {
            transfer.as_ref().and_then(lambert_delta_v)
        });
        decisions
            .iter()
            .zip(delta_vs)
            .map(|(decision, delta_v)| {
                delta_v.map(|value| evaluation_from_delta_v(*decision, value))
            })
            .collect()
    }

    /// Returns scalar objectives for an ordered staged batch.
    ///
    /// Invalid rows use the same finite replacement as [`Objective`].
    #[must_use]
    pub fn objective_batch(&self, decisions: &[Decision], workers: i32) -> Vec<f64> {
        self.evaluate_batch(decisions, workers)
            .into_iter()
            .map(|evaluation| evaluation.map_or(NAN_REPLACEMENT, |value| value.objective))
            .collect()
    }

    /// Returns the scalar objective or the optimizer's finite replacement.
    #[must_use]
    pub fn objective(&self, decision: Decision) -> f64 {
        self.evaluate(decision)
            .map_or(NAN_REPLACEMENT, |evaluation| evaluation.objective)
    }
}

fn valid_decision(decision: Decision) -> bool {
    decision.wait_days.is_finite()
        && decision.tof_days.is_finite()
        && (0.0..=MAX_WAIT_DAYS).contains(&decision.wait_days)
        && (MIN_TOF_DAYS..=MAX_TOF_DAYS).contains(&decision.tof_days)
}

fn evaluation_from_delta_v(decision: Decision, delta_v: f64) -> Evaluation {
    let departure_day = decision.wait_days;
    let arrival_day = departure_day + decision.tof_days;
    let excess_delta_v = (delta_v - DV_LIMIT).max(0.0);
    let objective = arrival_day
        + PENALTY_LINEAR * excess_delta_v
        + PENALTY_QUADRATIC * excess_delta_v * excess_delta_v;
    Evaluation {
        decision,
        departure_day,
        arrival_day,
        delta_v,
        excess_delta_v,
        objective,
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

    #[test]
    fn propagation_batch_matches_serial_and_preserves_order() {
        let problem = FixedLegProblem::easy().unwrap();
        let requests = [
            PropagationRequest::new(0, 0.0),
            PropagationRequest::new(1, 2.0),
            PropagationRequest::new(0, 4.0),
            PropagationRequest::new(2, 1.0),
            PropagationRequest::new(1, f64::NAN),
        ];
        let serial = requests
            .iter()
            .map(|request| problem.propagate(*request))
            .collect::<Vec<_>>();
        assert_eq!(problem.propagate_batch(&requests, 3), serial);
        assert!(serial[0].is_some());
        assert!(serial[3].is_none());
        assert!(serial[4].is_none());
    }

    #[test]
    fn lambert_batch_matches_serial_and_preserves_order() {
        let problem = FixedLegProblem::easy().unwrap();
        let mut transfers = [
            problem.transfer(Decision::new(0.0, 1.0)).unwrap(),
            problem.transfer(Decision::new(1.0, 2.0)).unwrap(),
            problem.transfer(Decision::new(2.0, 3.0)).unwrap(),
        ];
        transfers[1].tof_seconds = -1.0;
        let serial = transfers.iter().map(lambert_delta_v).collect::<Vec<_>>();
        assert_eq!(lambert_delta_v_batch(&transfers, 3), serial);
        assert!(serial[0].is_some());
        assert!(serial[1].is_none());
        assert!(serial[2].is_some());
    }

    #[test]
    fn staged_batch_matches_scalar_and_preserves_invalid_rows() {
        let problem = FixedLegProblem::easy().unwrap();
        let decisions = [
            Decision::new(0.0, 1.0),
            Decision::new(-1.0, 1.0),
            Decision::new(1.0, 2.0),
            Decision::new(12.0, 12.0),
            Decision::new(0.0, f64::NAN),
        ];
        let serial = decisions
            .iter()
            .map(|decision| problem.evaluate(*decision))
            .collect::<Vec<_>>();
        assert_eq!(problem.evaluate_batch(&decisions, 3), serial);
        assert_eq!(
            problem.objective_batch(&decisions, 3),
            serial
                .into_iter()
                .map(|value| value.map_or(NAN_REPLACEMENT, |result| result.objective))
                .collect::<Vec<_>>()
        );
    }
}
