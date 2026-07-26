// Copyright (c) 2026 pykep-lambert contributors
// SPDX-License-Identifier: MPL-2.0

//! Native fixed-leg objective and optimizer benchmark.

use std::env;
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;

use fcmaes_core::{BiteParams, Cmaes, CmaesParams, Fitness, optimize_bite};
use pykep_lambert_rust::{
    DV_LIMIT, DV_TOLERANCE, Decision, Evaluation, FixedLegProblem, MAX_TOF_DAYS, MAX_WAIT_DAYS,
    MIN_TOF_DAYS, SOURCE_COMMIT,
};

const DEFAULT_OBJECTIVE_EVALUATIONS: usize = 32_768;
const DEFAULT_OPTIMIZER_EVALUATIONS: u64 = 4_096;
const DEFAULT_CMA_POPULATION: i32 = 32;
const DEFAULT_SEED: u64 = 12_345;
const DEFAULT_WORKERS: i32 = 8;
const CMA_SIGMA: f64 = 0.25;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Optimizer {
    Cma,
    Bite,
    Both,
}

impl Optimizer {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cma" => Ok(Self::Cma),
            "bite" => Ok(Self::Bite),
            "both" => Ok(Self::Both),
            _ => Err("--optimizer must be cma, bite, or both".to_owned()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Args {
    objective_evaluations: usize,
    optimizer_evaluations: u64,
    optimizer: Optimizer,
    workers: i32,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            objective_evaluations: DEFAULT_OBJECTIVE_EVALUATIONS,
            optimizer_evaluations: DEFAULT_OPTIMIZER_EVALUATIONS,
            optimizer: Optimizer::Both,
            workers: DEFAULT_WORKERS,
            seed: DEFAULT_SEED,
        }
    }
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::parse_from(env::args().skip(1))
    }

    fn parse_from(arguments: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut parsed = Self::default();
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--objective-evaluations" => {
                    parsed.objective_evaluations =
                        parse_value(&mut arguments, "--objective-evaluations")?;
                }
                "--optimizer-evaluations" => {
                    parsed.optimizer_evaluations =
                        parse_value(&mut arguments, "--optimizer-evaluations")?;
                }
                "--optimizer" => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| "missing value after --optimizer".to_owned())?;
                    parsed.optimizer = Optimizer::parse(&value)?;
                }
                "--workers" => parsed.workers = parse_value(&mut arguments, "--workers")?,
                "--seed" => parsed.seed = parse_value(&mut arguments, "--seed")?,
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        if parsed.objective_evaluations == 0
            || parsed.optimizer_evaluations == 0
            || parsed.workers <= 0
        {
            return Err("evaluation counts and workers must be positive".to_owned());
        }
        Ok(parsed)
    }
}

fn parse_value<T: std::str::FromStr>(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    arguments
        .next()
        .ok_or_else(|| format!("missing value after {option}"))?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

fn print_help() {
    println!(
        "KTTSP Lambert-leg optimization benchmark\n\
         \nUsage: cargo run --release -- [OPTIONS]\n\
         \n  --objective-evaluations N  Raw objective calls (32768)\n\
         \n  --optimizer-evaluations N  Budget per optimizer (4096)\n\
         \n  --optimizer NAME           cma, bite, or both (both)\n\
         \n  --workers N                CMA objective workers (8)\n\
         \n  --seed N                   Deterministic optimizer/sample seed (12345)"
    );
}

#[derive(Clone, Debug)]
struct OptimizationReport {
    name: &'static str,
    seconds: f64,
    evaluations: u64,
    stop: i32,
    evaluation: Evaluation,
}

fn benchmark_objective(problem: &FixedLegProblem, evaluations: usize, seed: u64) {
    let _ = black_box(problem.objective(sample_decision(0, seed)));
    let started = Instant::now();
    let mut checksum = 0.0;
    let mut best = f64::INFINITY;
    for index in 0..evaluations {
        let value = black_box(problem.objective(sample_decision(index as u64, seed)));
        checksum += value;
        best = best.min(value);
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "OBJECTIVE evaluations={} best={:.12} checksum={:.12e} seconds={:.6} \
         evaluations_per_second={:.3}",
        evaluations,
        best,
        checksum,
        seconds,
        evaluations as f64 / seconds
    );
}

fn run_cma(
    problem: &FixedLegProblem,
    evaluations: u64,
    workers: i32,
    seed: u64,
) -> OptimizationReport {
    let lower = [0.0, MIN_TOF_DAYS];
    let upper = [MAX_WAIT_DAYS, MAX_TOF_DAYS];
    let initial = midpoint();
    let mut fitness = Fitness::bounded(2, 1, &lower, &upper);
    fitness.set_normalize(true);
    let params = CmaesParams {
        popsize: DEFAULT_CMA_POPULATION,
        max_evaluations: evaluations,
        seed,
        ..Default::default()
    };
    let mut optimizer = Cmaes::new(fitness, &initial, &[CMA_SIGMA], &params);
    let started = Instant::now();
    let result = optimizer.optimize(problem, workers);
    let seconds = started.elapsed().as_secs_f64();
    let evaluation = problem
        .evaluate(Decision::new(result.x[0], result.x[1]))
        .expect("CMA result must remain inside validated bounds");
    OptimizationReport {
        name: "cma",
        seconds,
        evaluations: result.evaluations,
        stop: result.stop,
        evaluation,
    }
}

fn run_bite(problem: &FixedLegProblem, evaluations: u64, seed: u64) -> OptimizationReport {
    let lower = [0.0, MIN_TOF_DAYS];
    let upper = [MAX_WAIT_DAYS, MAX_TOF_DAYS];
    let initial = midpoint();
    let params = BiteParams {
        max_evaluations: evaluations,
        seed,
        ..Default::default()
    };
    let started = Instant::now();
    let result = optimize_bite(problem, &lower, &upper, Some(&initial), &params, 1);
    let seconds = started.elapsed().as_secs_f64();
    let evaluation = problem
        .evaluate(Decision::new(result.x[0], result.x[1]))
        .expect("BiteOpt result must remain inside validated bounds");
    OptimizationReport {
        name: "bite",
        seconds,
        evaluations: result.evaluations,
        stop: result.stop,
        evaluation,
    }
}

fn midpoint() -> [f64; 2] {
    [MAX_WAIT_DAYS / 2.0, (MIN_TOF_DAYS + MAX_TOF_DAYS) / 2.0]
}

fn print_optimizer(report: &OptimizationReport) {
    let evaluation = report.evaluation;
    println!(
        "OPTIMIZER name={} evaluations={} stop={} objective={:.12} wait_days={:.12} \
         tof_days={:.12} arrival_day={:.12} delta_v={:.9} feasible={} seconds={:.6} \
         evaluations_per_second={:.3}",
        report.name,
        report.evaluations,
        report.stop,
        evaluation.objective,
        evaluation.decision.wait_days,
        evaluation.decision.tof_days,
        evaluation.arrival_day,
        evaluation.delta_v,
        evaluation.excess_delta_v <= DV_TOLERANCE,
        report.seconds,
        report.evaluations as f64 / report.seconds
    );
}

fn sample_decision(index: u64, seed: u64) -> Decision {
    let wait =
        unit_interval(splitmix64(seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15))) * MAX_WAIT_DAYS;
    let tof_unit = unit_interval(splitmix64(
        seed ^ index.wrapping_mul(0xbf58_476d_1ce4_e5b9) ^ 0x94d0_49bb_1331_11eb,
    ));
    Decision::new(
        wait,
        MIN_TOF_DAYS + tof_unit * (MAX_TOF_DAYS - MIN_TOF_DAYS),
    )
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn unit_interval(value: u64) -> f64 {
    (value >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    let problem = FixedLegProblem::easy()?;
    println!(
        "CONFIG source_commit={} pykep_core=0.1.1 fcmaes_core=0.1.3 \
         objective_evaluations={} optimizer_evaluations={} optimizer={:?} \
         cma_population={} cma_workers={} seed={} dv_limit={}",
        SOURCE_COMMIT,
        args.objective_evaluations,
        args.optimizer_evaluations,
        args.optimizer,
        DEFAULT_CMA_POPULATION,
        args.workers,
        args.seed,
        DV_LIMIT
    );
    benchmark_objective(&problem, args.objective_evaluations, args.seed);
    if matches!(args.optimizer, Optimizer::Cma | Optimizer::Both) {
        print_optimizer(&run_cma(
            &problem,
            args.optimizer_evaluations,
            args.workers,
            args.seed,
        ));
    }
    if matches!(args.optimizer, Optimizer::Bite | Optimizer::Both) {
        print_optimizer(&run_bite(&problem, args.optimizer_evaluations, args.seed));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_source_workload() {
        let args = Args::default();
        assert_eq!(args.objective_evaluations, 32_768);
        assert_eq!(args.optimizer_evaluations, 4_096);
        assert_eq!(args.optimizer, Optimizer::Both);
        assert_eq!(DEFAULT_CMA_POPULATION, 32);
        assert_eq!(args.seed, 12_345);
    }

    #[test]
    fn arguments_are_validated() {
        let args = Args::parse_from([
            "--objective-evaluations".to_owned(),
            "128".to_owned(),
            "--optimizer".to_owned(),
            "bite".to_owned(),
            "--workers".to_owned(),
            "2".to_owned(),
        ])
        .unwrap();
        assert_eq!(args.objective_evaluations, 128);
        assert_eq!(args.optimizer, Optimizer::Bite);
        assert_eq!(args.workers, 2);
        assert!(Args::parse_from(["--workers".to_owned(), "0".to_owned()]).is_err());
        assert!(Args::parse_from(["--optimizer".to_owned(), "de".to_owned()]).is_err());
    }

    #[test]
    fn deterministic_candidates_stay_in_bounds() {
        for index in 0..1_000 {
            let decision = sample_decision(index, DEFAULT_SEED);
            assert!((0.0..=MAX_WAIT_DAYS).contains(&decision.wait_days));
            assert!((MIN_TOF_DAYS..=MAX_TOF_DAYS).contains(&decision.tof_days));
            assert_eq!(decision, sample_decision(index, DEFAULT_SEED));
        }
    }

    #[test]
    fn both_fcmaes_optimizers_complete_a_smoke_budget() {
        let problem = FixedLegProblem::easy().unwrap();
        for report in [
            run_cma(&problem, 256, 2, DEFAULT_SEED),
            run_bite(&problem, 256, DEFAULT_SEED),
        ] {
            assert!(report.evaluations > 0);
            assert!(report.evaluation.objective.is_finite());
            assert!(report.evaluation.delta_v.is_finite());
        }
    }
}
