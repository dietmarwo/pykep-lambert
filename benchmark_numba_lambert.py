#!/usr/bin/env python3
"""Benchmark pykep Lambert against the local Numba port.

The benchmark covers:
  * pykep Lambert API calls on pre-propagated states
  * full KTTSP ``compute_transfer`` calls, including ephemeris propagation
  * a small fcmaes CMA-ES ask/tell optimization demo using batched transfer
    evaluation

Example:
    python benchmark_numba_lambert.py
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pykep as pk
from fcmaes import bitecpp
from fcmaes import cmaescpp
from numba import config as numba_config
from numba import get_num_threads, set_num_threads
from scipy.optimize import Bounds

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

DEFAULT_PROBLEM = SCRIPT_DIR / "problems" / "easy.kttsp"
DEFAULT_JSON_OUT = SCRIPT_DIR / "results" / "benchmark_latest.json"

import pk_lambert as pl
from numba_transfer import NumbaTransfer
from utils_users import TomatoProblem


DAY = 86400.0

_WORKER_PROBLEM: TomatoProblem | None = None
_WORKER_TABLE_T_START_GRIDS = None
_WORKER_TABLE_TOF_GRIDS = None
_WORKER_TABLE_DV_LIMIT = None
_WORKER_TABLE_TARGET_MAX_TIME = None
_WORKER_TABLE_STORE_MODE = None


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(resolved)


def _max_revs_arg(values: list[int]) -> int | list[int]:
    if len(values) == 1:
        return int(values[0])
    return [int(v) for v in values]


def _set_threads(requested: int) -> tuple[int, int]:
    old = int(get_num_threads())
    max_threads = int(getattr(numba_config, "NUMBA_NUM_THREADS", old))
    target = max(1, min(int(requested), max_threads))
    set_num_threads(target)
    return old, target


def _time_call(fn):
    gc.collect()
    t0 = time.perf_counter()
    value = fn()
    return time.perf_counter() - t0, value


def _pykep_lambert_dv(case) -> float:
    r1, v_dep, r2, v_arr, tof_sec, mu, max_revs = case
    best = float("inf")
    for cw in (False, True):
        try:
            lambert_solution = pk.lambert_problem(r1, r2, float(tof_sec), float(mu), cw, int(max_revs))
        except Exception:
            continue
        for v1, v2 in zip(lambert_solution.get_v1(), lambert_solution.get_v2()):
            dv1 = np.linalg.norm(np.asarray(v1, dtype=float) - np.asarray(v_dep, dtype=float))
            dv2 = np.linalg.norm(np.asarray(v2, dtype=float) - np.asarray(v_arr, dtype=float))
            best = min(best, float(dv1 + dv2))
    return best


def _init_problem_worker(problem_path: str, max_revs):
    global _WORKER_PROBLEM
    _WORKER_PROBLEM = TomatoProblem(problem_path, max_revs=max_revs)


def _pykep_compute_transfer_worker(row) -> float:
    if _WORKER_PROBLEM is None:
        raise RuntimeError("worker problem was not initialized")
    i, j, t_start, tof = row
    return float(_WORKER_PROBLEM.compute_transfer(int(i), int(j), float(t_start), float(tof)))


def _init_pykep_table_worker(
    problem_path: str,
    max_revs,
    t_start_grids,
    tof_grids,
    dv_limit,
    target_max_time: float,
    store_mode: str,
):
    global _WORKER_PROBLEM, _WORKER_TABLE_T_START_GRIDS, _WORKER_TABLE_TOF_GRIDS
    global _WORKER_TABLE_DV_LIMIT, _WORKER_TABLE_TARGET_MAX_TIME, _WORKER_TABLE_STORE_MODE
    _WORKER_PROBLEM = TomatoProblem(problem_path, max_revs=max_revs)
    _WORKER_TABLE_T_START_GRIDS = t_start_grids
    _WORKER_TABLE_TOF_GRIDS = tof_grids
    _WORKER_TABLE_DV_LIMIT = np.asarray(dv_limit, dtype=np.float64)
    _WORKER_TABLE_TARGET_MAX_TIME = float(target_max_time)
    _WORKER_TABLE_STORE_MODE = str(store_mode)


def _pykep_table_warm_worker(_worker_index: int) -> int:
    if _WORKER_PROBLEM is None:
        raise RuntimeError("worker problem was not initialized")
    try:
        _WORKER_PROBLEM.compute_transfer(0, 1, 0.0, max(float(_WORKER_PROBLEM.min_tof), 1.0))
    except Exception:
        pass
    return 0


def _pykep_table_pair_worker(task):
    if _WORKER_PROBLEM is None:
        raise RuntimeError("worker problem was not initialized")
    pair_index, i_from, i_to = task
    transfer_type = int(_WORKER_PROBLEM.transfer_type(int(i_from), int(i_to)))
    dep_grid = _WORKER_TABLE_T_START_GRIDS[transfer_type]
    tof_grid = _WORKER_TABLE_TOF_GRIDS[transfer_type]
    dv_limit = float(_WORKER_TABLE_DV_LIMIT[transfer_type])
    target_max_time = float(_WORKER_TABLE_TARGET_MAX_TIME)
    store_mode = str(_WORKER_TABLE_STORE_MODE)

    evaluated = 0
    retained_all = 0
    best_by_dep = np.full(len(dep_grid), np.inf, dtype=np.float64)

    for dep_pos, dep in enumerate(dep_grid):
        dep_f = float(dep)
        for tof in tof_grid:
            tof_f = float(tof)
            if dep_f + tof_f > target_max_time + 1.0e-12:
                continue
            evaluated += 1
            dv = float(_WORKER_PROBLEM.compute_transfer(int(i_from), int(i_to), dep_f, tof_f))
            if not math.isfinite(dv) or dv > dv_limit:
                continue
            if store_mode == "all":
                retained_all += 1
            elif store_mode == "best-per-departure":
                if dv < best_by_dep[dep_pos]:
                    best_by_dep[dep_pos] = dv
            else:
                raise ValueError(f"unsupported table store mode: {store_mode}")

    retained = retained_all if store_mode == "all" else int(np.sum(np.isfinite(best_by_dep)))
    return int(pair_index), int(evaluated), int(retained), int(retained > 0), int(transfer_type)


def _pool_map(function, rows, workers: int, initializer=None, initargs=()):
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
    chunksize = max(1, len(rows) // max(1, int(workers) * 8))
    with ctx.Pool(processes=int(workers), initializer=initializer, initargs=initargs) as pool:
        # Warm each process before timing so the reported batch number is the
        # steady-state map, not pool construction/import latency.
        warm = rows[: max(1, min(len(rows), int(workers)))]
        list(pool.map(function, warm, chunksize=1))
        dt, values = _time_call(lambda: list(pool.map(function, rows, chunksize=chunksize)))
    return dt, np.asarray(values, dtype=np.float64)


def _generate_transfer_rows(problem: TomatoProblem, samples: int, tof_max: float, seed: int):
    rng = np.random.default_rng(int(seed))
    n = int(problem.n_tomatoes)
    i_idx = rng.integers(0, n, size=int(samples), dtype=np.int64)
    j_idx = rng.integers(0, n - 1, size=int(samples), dtype=np.int64)
    j_idx = np.where(j_idx >= i_idx, j_idx + 1, j_idx).astype(np.int64)
    t_hi = max(float(problem.min_tof), float(problem.max_time) - float(tof_max) - 1.0)
    t_start = rng.uniform(0.0, t_hi, size=int(samples))
    tof = rng.uniform(float(problem.min_tof), float(tof_max), size=int(samples))
    return [
        (int(i), int(j), float(t), float(tf))
        for i, j, t, tf in zip(i_idx, j_idx, t_start, tof)
    ]


def _prepare_lambert_cases(problem: TomatoProblem, numba_transfer: NumbaTransfer, rows):
    r1 = np.empty((len(rows), 3), dtype=np.float64)
    v_dep = np.empty((len(rows), 3), dtype=np.float64)
    r2 = np.empty((len(rows), 3), dtype=np.float64)
    v_arr = np.empty((len(rows), 3), dtype=np.float64)
    tof_sec = np.empty(len(rows), dtype=np.float64)
    max_revs = np.empty(len(rows), dtype=np.int32)

    cases = []
    for k, (i, j, t_start, tof) in enumerate(rows):
        ri, vi = problem.tomatoes[int(i)].eph(float(t_start))
        rj, vj = problem.tomatoes[int(j)].eph(float(t_start + tof))
        r1[k] = np.asarray(ri, dtype=np.float64)
        v_dep[k] = np.asarray(vi, dtype=np.float64)
        r2[k] = np.asarray(rj, dtype=np.float64)
        v_arr[k] = np.asarray(vj, dtype=np.float64)
        tof_sec[k] = float(tof) * DAY
        max_revs[k] = int(numba_transfer.max_revs_pair[int(i), int(j)])
        cases.append((r1[k], v_dep[k], r2[k], v_arr[k], tof_sec[k], numba_transfer.mu, int(max_revs[k])))
    return cases, r1, v_dep, r2, v_arr, tof_sec, max_revs


def _warm_numba(numba_transfer: NumbaTransfer, rows, lambert_arrays):
    old, _target = _set_threads(1)
    try:
        first = rows[0]
        numba_transfer.compute_transfer(*first)
        numba_transfer.compute_transfer_batch(rows[: min(32, len(rows))])
        _cases, r1, v_dep, r2, v_arr, tof_sec, max_revs = lambert_arrays
        pl.lambert_dv(r1[0], v_dep[0], r2[0], v_arr[0], tof_sec[0], numba_transfer.mu, int(max_revs[0]))
        out = np.empty(min(32, len(rows)), dtype=np.float64)
        pl.lambert_dv_batch(
            r1[: out.size],
            v_dep[: out.size],
            r2[: out.size],
            v_arr[: out.size],
            tof_sec[: out.size],
            numba_transfer.mu,
            max_revs[: out.size],
            out,
        )
        numba_transfer.fixed_leg_objective_batch(
            0,
            1,
            0.0,
            np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64),
            min_tof=0.001,
            max_wait=12.0,
            max_tof=12.0,
            dv_limit=600.0,
            penalty_linear=1000.0,
            penalty_quad=1.0,
        )
        numba_transfer.fixed_leg_objective_one(
            0,
            1,
            0.0,
            0.0,
            1.0,
            min_tof=0.001,
            max_wait=12.0,
            max_tof=12.0,
            dv_limit=600.0,
            penalty_linear=1000.0,
            penalty_quad=1.0,
        )
    finally:
        set_num_threads(old)


def _benchmark_lambert(args, problem, numba_transfer, lambert_arrays):
    cases, r1, v_dep, r2, v_arr, tof_sec, max_revs = lambert_arrays
    n = len(cases)

    dt_pykep_serial, pykep_serial = _time_call(
        lambda: np.asarray([_pykep_lambert_dv(case) for case in cases], dtype=np.float64)
    )
    dt_pykep_batch, pykep_batch = _pool_map(_pykep_lambert_dv, cases, args.workers)

    old, _ = _set_threads(1)
    try:
        dt_numba_serial, numba_serial = _time_call(
            lambda: np.asarray(
                [
                    pl.lambert_dv(
                        r1[k],
                        v_dep[k],
                        r2[k],
                        v_arr[k],
                        tof_sec[k],
                        numba_transfer.mu,
                        int(max_revs[k]),
                    )
                    for k in range(n)
                ],
                dtype=np.float64,
            )
        )
    finally:
        set_num_threads(old)

    old, numba_threads = _set_threads(args.workers)
    try:
        out = np.empty(n, dtype=np.float64)
        dt_numba_batch, _ = _time_call(
            lambda: pl.lambert_dv_batch(
                r1,
                v_dep,
                r2,
                v_arr,
                tof_sec,
                numba_transfer.mu,
                max_revs,
                out,
            )
        )
        numba_batch = out.copy()
    finally:
        set_num_threads(old)

    return {
        "numba_threads": numba_threads,
        "results": [
            _bench_row("lambert", "pykep serial", n, dt_pykep_serial, 1.0),
            _bench_row("lambert", f"pykep batch {args.workers}", n, dt_pykep_batch, dt_pykep_serial / dt_pykep_batch),
            _bench_row("lambert", "numba serial", n, dt_numba_serial, dt_pykep_serial / dt_numba_serial),
            _bench_row("lambert", f"numba batch {numba_threads}", n, dt_numba_batch, dt_pykep_serial / dt_numba_batch),
        ],
        "correctness": _correctness(pykep_serial, numba_serial, extra={"pykep_batch_max_abs": _max_abs(pykep_serial, pykep_batch), "numba_batch_max_abs": _max_abs(numba_serial, numba_batch)}),
    }


def _benchmark_compute_transfer(args, problem, numba_transfer, rows):
    n = len(rows)

    dt_pykep_serial, pykep_serial = _time_call(
        lambda: np.asarray([problem.compute_transfer(*row) for row in rows], dtype=np.float64)
    )
    dt_pykep_batch, pykep_batch = _pool_map(
        _pykep_compute_transfer_worker,
        rows,
        args.workers,
        initializer=_init_problem_worker,
        initargs=(str(args.problem), _max_revs_arg(args.max_revs)),
    )

    old, _ = _set_threads(1)
    try:
        dt_numba_serial, numba_serial = _time_call(
            lambda: np.asarray([numba_transfer.compute_transfer(*row) for row in rows], dtype=np.float64)
        )
    finally:
        set_num_threads(old)

    old, numba_threads = _set_threads(args.workers)
    try:
        dt_numba_batch, numba_batch_values = _time_call(
            lambda: np.asarray(numba_transfer.compute_transfer_batch(rows), dtype=np.float64)
        )
    finally:
        set_num_threads(old)

    return {
        "numba_threads": numba_threads,
        "results": [
            _bench_row("compute_transfer", "pykep serial", n, dt_pykep_serial, 1.0),
            _bench_row("compute_transfer", f"pykep batch {args.workers}", n, dt_pykep_batch, dt_pykep_serial / dt_pykep_batch),
            _bench_row("compute_transfer", "numba serial", n, dt_numba_serial, dt_pykep_serial / dt_numba_serial),
            _bench_row("compute_transfer", f"numba batch {numba_threads}", n, dt_numba_batch, dt_pykep_serial / dt_numba_batch),
        ],
        "correctness": _correctness(pykep_serial, numba_serial, extra={"pykep_batch_max_abs": _max_abs(pykep_serial, pykep_batch), "numba_batch_max_abs": _max_abs(numba_serial, numba_batch_values)}),
    }


def _bench_row(section: str, mode: str, samples: int, seconds: float, speedup: float):
    return {
        "section": section,
        "mode": mode,
        "samples": int(samples),
        "seconds": float(seconds),
        "calls_per_second": float(samples / seconds),
        "speedup_vs_pykep_serial": float(speedup),
    }


def _max_abs(a, b) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    return float(np.max(np.abs(a[finite] - b[finite])))


def _correctness(reference, candidate, extra=None):
    finite = np.isfinite(reference) & np.isfinite(candidate)
    out = {
        "finite_compared": int(np.sum(finite)),
        "max_abs_dv_m_s": _max_abs(reference, candidate),
        "mean_abs_dv_m_s": float(np.mean(np.abs(reference[finite] - candidate[finite]))) if np.any(finite) else float("nan"),
        "max_rel": float(np.max(np.abs(reference[finite] - candidate[finite]) / np.maximum(1.0, np.abs(reference[finite])))) if np.any(finite) else float("nan"),
    }
    if extra:
        out.update(extra)
    return out


def _triplet(values, dtype=float) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.size == 1:
        return np.full(3, arr[0], dtype=dtype)
    if arr.size != 3:
        raise ValueError(f"expected scalar or 3 values, got {arr.size}")
    return arr.astype(dtype, copy=False)


def _plane_normal(inclination: float, raan: float) -> np.ndarray:
    sin_i = np.sin(float(inclination))
    return np.asarray(
        [
            sin_i * np.sin(float(raan)),
            -sin_i * np.cos(float(raan)),
            np.cos(float(inclination)),
        ],
        dtype=np.float64,
    )


def _plane_angle(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(normal_a, normal_b)), -1.0, 1.0))
    return float(np.arccos(cosine))


def _estimate_minimum_transfer_dv(problem, i_from: int, i_to: int, plane_normals, circular_speeds):
    rows = problem.tomato_orbital_parameters
    transfer_type = int(problem.transfer_type(i_from, i_to))
    delta_plane = _plane_angle(plane_normals[i_from], plane_normals[i_to])

    if transfer_type != 2:
        speed = float(min(circular_speeds[i_from], circular_speeds[i_to]))
        return float(2.0 * speed * np.sin(0.5 * delta_plane)), delta_plane

    r_inner = float(min(rows[i_from, 0], rows[i_to, 0]))
    r_outer = float(max(rows[i_from, 0], rows[i_to, 0]))
    transfer_a = 0.5 * (r_inner + r_outer)
    mu = float(problem.mu_moon)

    v_inner = np.sqrt(mu / r_inner)
    v_outer = np.sqrt(mu / r_outer)
    v_transfer_inner = np.sqrt(mu * (2.0 / r_inner - 1.0 / transfer_a))
    v_transfer_outer = np.sqrt(mu * (2.0 / r_outer - 1.0 / transfer_a))
    burn_inner = abs(v_transfer_inner - v_inner)
    burn_outer = np.sqrt(
        v_transfer_outer * v_transfer_outer
        + v_outer * v_outer
        - 2.0 * v_transfer_outer * v_outer * np.cos(delta_plane)
    )
    return float(burn_inner + burn_outer), delta_plane


def _table_pair_matches(problem, i_from: int, i_to: int, cluster: str) -> bool:
    if cluster == "all":
        return True
    if cluster == "inner":
        return bool(problem.is_inner(i_from, i_to))
    if cluster == "outer":
        return bool(problem.is_outer(i_from, i_to))
    if cluster == "switch":
        return bool(problem.is_cluster_switch(i_from, i_to))
    raise ValueError(f"unsupported table cluster: {cluster}")


def _table_pairs(problem, cluster: str, dv_limit: np.ndarray):
    rows = np.asarray(problem.tomato_orbital_parameters, dtype=np.float64)
    plane_normals = np.asarray([_plane_normal(row[2], row[3]) for row in rows], dtype=np.float64)
    circular_speeds = np.sqrt(float(problem.mu_moon) / rows[:, 0])
    pairs = []
    stats = {"rejected_self": 0, "rejected_cluster": 0, "rejected_dv_bound": 0}
    for i in range(int(problem.n_tomatoes)):
        for j in range(int(problem.n_tomatoes)):
            if i == j:
                stats["rejected_self"] += 1
                continue
            if not _table_pair_matches(problem, i, j, cluster):
                stats["rejected_cluster"] += 1
                continue
            transfer_type = int(problem.transfer_type(i, j))
            min_dv, _delta_plane = _estimate_minimum_transfer_dv(
                problem,
                i,
                j,
                plane_normals,
                circular_speeds,
            )
            if float(min_dv) > float(dv_limit[transfer_type]):
                stats["rejected_dv_bound"] += 1
                continue
            pairs.append((i, j))
    stats["kept"] = int(len(pairs))
    return np.asarray(pairs, dtype=np.int32).reshape((-1, 2)), stats


def _table_grids(problem, min_tof, max_tof, tof_step, t_start_step, target_max_time):
    target = float(problem.max_time) if target_max_time is None else float(target_max_time)
    if target <= 0.0 or target > float(problem.max_time) + 1.0e-12:
        raise ValueError(f"table target max time must be in (0, {float(problem.max_time):.6g}]")

    t_start_grids = []
    tof_grids = []
    for transfer_type in range(3):
        dep_step = float(t_start_step[transfer_type])
        tf_step = float(tof_step[transfer_type])
        if dep_step <= 0.0 or tf_step <= 0.0:
            raise ValueError("table grid steps must be positive")
        stop = min(target, float(problem.max_time)) - float(min_tof[transfer_type])
        if stop <= 0.0:
            t_start_grids.append(np.empty(0, dtype=np.float64))
        else:
            t_start_grids.append(np.arange(0.0, stop, dep_step, dtype=np.float64))
        tof_grids.append(
            np.arange(
                float(min_tof[transfer_type]),
                float(max_tof[transfer_type]) + 0.5 * tf_step,
                tf_step,
                dtype=np.float64,
            )
        )
    return t_start_grids, tof_grids, target


def _count_table_cells(problem, pairs, t_start_grids, tof_grids, target_max_time: float) -> int:
    total = 0
    for i, j in np.asarray(pairs, dtype=np.int32).reshape((-1, 2)):
        transfer_type = int(problem.transfer_type(int(i), int(j)))
        dep_grid = t_start_grids[transfer_type]
        tof_grid = tof_grids[transfer_type]
        if dep_grid.size == 0 or tof_grid.size == 0:
            continue
        dep_idx = np.repeat(np.arange(dep_grid.size, dtype=np.int64), tof_grid.size)
        tof_idx = np.tile(np.arange(tof_grid.size, dtype=np.int64), dep_grid.size)
        total += int(np.sum(dep_grid[dep_idx] + tof_grid[tof_idx] <= float(target_max_time) + 1.0e-12))
    return total


def _evaluate_table_pair(numba_transfer, i: int, j: int, dep_grid, tof_grid, target_max_time: float, dv_limit: float, store_mode: str, chunk_size: int):
    dep_count = int(dep_grid.size)
    tof_count = int(tof_grid.size)
    total_rectangular = dep_count * tof_count
    if total_rectangular == 0:
        return 0, 0, 0

    evaluated = 0
    retained_all = 0
    best_by_dep = np.full(dep_count, np.inf, dtype=np.float64)
    chunk_size = max(1, int(chunk_size))

    for start in range(0, total_rectangular, chunk_size):
        stop = min(total_rectangular, start + chunk_size)
        flat = np.arange(start, stop, dtype=np.int64)
        dep_idx = flat // tof_count
        tof_idx = flat - dep_idx * tof_count
        dep = dep_grid[dep_idx]
        tof = tof_grid[tof_idx]
        within_mission = dep + tof <= float(target_max_time) + 1.0e-12
        if not np.any(within_mission):
            continue
        dep_idx = dep_idx[within_mission]
        dep = np.ascontiguousarray(dep[within_mission])
        tof = np.ascontiguousarray(tof[within_mission])
        n = int(dep.size)
        out = np.empty(n, dtype=np.float64)
        pl.dv_full_batch(
            numba_transfer.A,
            numba_transfer.E,
            numba_transfer.I,
            numba_transfer.O,
            numba_transfer.Oc,
            numba_transfer.M0,
            numba_transfer.NN,
            numba_transfer.mu,
            np.full(n, int(i), dtype=np.int64),
            np.full(n, int(j), dtype=np.int64),
            np.ascontiguousarray(dep * DAY),
            np.ascontiguousarray(tof * DAY),
            numba_transfer.max_revs_pair,
            out,
        )
        evaluated += n
        keep = np.isfinite(out) & (out <= float(dv_limit))
        if not np.any(keep):
            continue
        if store_mode == "all":
            retained_all += int(np.sum(keep))
        elif store_mode == "best-per-departure":
            np.minimum.at(best_by_dep, dep_idx[keep], out[keep])
        else:
            raise ValueError(f"unsupported table store mode: {store_mode}")

    retained = retained_all if store_mode == "all" else int(np.sum(np.isfinite(best_by_dep)))
    nonempty = int(retained > 0)
    return evaluated, retained, nonempty


def _run_pykep_table_creation(args, problem, pairs, t_start_grids, tof_grids, target_max_time, dv_limit):
    tasks = [
        (int(pair_index), int(i), int(j))
        for pair_index, (i, j) in enumerate(np.asarray(pairs, dtype=np.int32).reshape((-1, 2)))
    ]
    workers = max(1, int(args.table_pykep_workers))
    initargs = (
        str(args.problem),
        _max_revs_arg(args.max_revs),
        t_start_grids,
        tof_grids,
        np.asarray(dv_limit, dtype=np.float64),
        float(target_max_time),
        str(args.table_store_mode),
    )

    def aggregate(results):
        evaluated = 0
        retained = 0
        nonempty_pairs = 0
        type_counts = np.zeros(3, dtype=np.int64)
        retained_by_type = np.zeros(3, dtype=np.int64)
        evaluated_by_type = np.zeros(3, dtype=np.int64)
        for _pair_index, pair_evaluated, pair_retained, pair_nonempty, transfer_type in results:
            ttype = int(transfer_type)
            type_counts[ttype] += 1
            evaluated += int(pair_evaluated)
            retained += int(pair_retained)
            nonempty_pairs += int(pair_nonempty)
            evaluated_by_type[ttype] += int(pair_evaluated)
            retained_by_type[ttype] += int(pair_retained)
        return {
            "evaluated_cells": int(evaluated),
            "retained_cells": int(retained),
            "nonempty_pairs": int(nonempty_pairs),
            "pair_type_counts": type_counts.tolist(),
            "evaluated_by_type": evaluated_by_type.tolist(),
            "retained_by_type": retained_by_type.tolist(),
        }

    if workers == 1:
        _init_pykep_table_worker(*initargs)
        _pykep_table_warm_worker(0)
        seconds, stats = _time_call(lambda: aggregate(_pykep_table_pair_worker(task) for task in tasks))
    else:
        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
        with ctx.Pool(processes=workers, initializer=_init_pykep_table_worker, initargs=initargs) as pool:
            list(pool.map(_pykep_table_warm_worker, range(workers), chunksize=1))
            seconds, stats = _time_call(
                lambda: aggregate(
                    pool.imap_unordered(
                        _pykep_table_pair_worker,
                        tasks,
                        chunksize=max(1, int(args.table_pykep_chunksize)),
                    )
                )
            )

    return {
        "section": "table_creation",
        "mode": f"pykep table batch {workers}",
        "samples": int(stats["evaluated_cells"]),
        "seconds": float(seconds),
        "calls_per_second": float(stats["evaluated_cells"] / seconds),
        "speedup_vs_pykep_serial": None,
        "retained_cells": int(stats["retained_cells"]),
        "nonempty_pairs": int(stats["nonempty_pairs"]),
        "pair_count": int(len(pairs)),
        "pair_type_counts": stats["pair_type_counts"],
        "evaluated_by_type": stats["evaluated_by_type"],
        "retained_by_type": stats["retained_by_type"],
    }


def _benchmark_table_creation(args, problem, numba_transfer):
    min_tof = _triplet(args.table_min_tof, dtype=np.float64)
    max_tof = _triplet(args.table_max_tof, dtype=np.float64)
    tof_step = _triplet(args.table_tof_step, dtype=np.float64)
    t_start_step = _triplet(args.table_t_start_step, dtype=np.float64)
    dv_limit = _triplet(args.table_dv_limit, dtype=np.float64)
    pairs, pair_stats = _table_pairs(problem, str(args.table_cluster), dv_limit)
    t_start_grids, tof_grids, target_max_time = _table_grids(
        problem,
        min_tof,
        max_tof,
        tof_step,
        t_start_step,
        None if args.table_target_max_time is None else float(args.table_target_max_time),
    )
    expected_cells = _count_table_cells(problem, pairs, t_start_grids, tof_grids, target_max_time)

    old, numba_threads = _set_threads(int(args.table_workers))
    try:
        def run():
            evaluated = 0
            retained = 0
            nonempty_pairs = 0
            type_counts = np.zeros(3, dtype=np.int64)
            retained_by_type = np.zeros(3, dtype=np.int64)
            evaluated_by_type = np.zeros(3, dtype=np.int64)
            for i, j in pairs:
                transfer_type = int(problem.transfer_type(int(i), int(j)))
                type_counts[transfer_type] += 1
                pair_evaluated, pair_retained, pair_nonempty = _evaluate_table_pair(
                    numba_transfer,
                    int(i),
                    int(j),
                    t_start_grids[transfer_type],
                    tof_grids[transfer_type],
                    target_max_time,
                    float(dv_limit[transfer_type]),
                    str(args.table_store_mode),
                    int(args.table_batch_chunk),
                )
                evaluated += int(pair_evaluated)
                retained += int(pair_retained)
                nonempty_pairs += int(pair_nonempty)
                evaluated_by_type[transfer_type] += int(pair_evaluated)
                retained_by_type[transfer_type] += int(pair_retained)
            return {
                "evaluated_cells": int(evaluated),
                "retained_cells": int(retained),
                "nonempty_pairs": int(nonempty_pairs),
                "pair_type_counts": type_counts.tolist(),
                "evaluated_by_type": evaluated_by_type.tolist(),
                "retained_by_type": retained_by_type.tolist(),
            }

        seconds, stats = _time_call(run)
    finally:
        set_num_threads(old)

    numba_row = {
        "section": "table_creation",
        "mode": f"numba table batch {numba_threads}",
        "samples": int(stats["evaluated_cells"]),
        "seconds": float(seconds),
        "calls_per_second": float(stats["evaluated_cells"] / seconds),
        "speedup_vs_pykep_serial": None,
        "retained_cells": int(stats["retained_cells"]),
        "nonempty_pairs": int(stats["nonempty_pairs"]),
        "pair_count": int(len(pairs)),
        "pair_type_counts": stats["pair_type_counts"],
        "evaluated_by_type": stats["evaluated_by_type"],
        "retained_by_type": stats["retained_by_type"],
    }

    results = [numba_row]
    pykep_workers = None
    if bool(args.include_pykep_table):
        pykep_row = _run_pykep_table_creation(
            args,
            problem,
            pairs,
            t_start_grids,
            tof_grids,
            target_max_time,
            dv_limit,
        )
        pykep_workers = int(args.table_pykep_workers)
        if int(pykep_row["samples"]) != int(numba_row["samples"]):
            raise RuntimeError("pykep and numba table benchmarks evaluated different cell counts")
        numba_row["speedup_vs_pykep_table_batch"] = float(pykep_row["seconds"] / numba_row["seconds"])
        pykep_row["speedup_vs_pykep_table_batch"] = 1.0
        results = [pykep_row, numba_row]

    return {
        "numba_threads": int(numba_threads),
        "pykep_workers": pykep_workers,
        "settings": {
            "cluster": str(args.table_cluster),
            "target_max_time": float(target_max_time),
            "min_tof": min_tof.tolist(),
            "max_tof": max_tof.tolist(),
            "tof_step": tof_step.tolist(),
            "t_start_step": t_start_step.tolist(),
            "dv_limit": dv_limit.tolist(),
            "store_mode": str(args.table_store_mode),
            "batch_chunk": int(args.table_batch_chunk),
            "t_start_grid_sizes": [int(grid.size) for grid in t_start_grids],
            "tof_grid_sizes": [int(grid.size) for grid in tof_grids],
            "pair_filter": pair_stats,
            "expected_valid_cells": int(expected_cells),
        },
        "results": results,
    }


class TransferEvaluator:
    def __init__(self, problem, numba_transfer, backend: str):
        self.problem = problem
        self.numba_transfer = numba_transfer
        self.backend = backend

    def compute_batch(self, transfers):
        if self.backend == "pykep":
            return [
                float(self.problem.compute_transfer(int(i), int(j), float(t_start), float(tof)))
                for i, j, t_start, tof in transfers
            ]
        if self.backend == "numba":
            return self.numba_transfer.compute_transfer_batch(transfers)
        raise ValueError(f"unknown backend: {self.backend}")


def _record_best(info, *, best_seen, best_feasible, dv_tol: float):
    if not math.isfinite(float(info["dv"])):
        return best_seen, best_feasible
    if best_seen is None or (info["excess"], info["arrival"]) < (best_seen["excess"], best_seen["arrival"]):
        best_seen = dict(info)
    if info["excess"] <= float(dv_tol):
        if best_feasible is None or info["arrival"] < best_feasible["arrival"]:
            best_feasible = dict(info)
    return best_seen, best_feasible


def _optimize_leg_ask_tell(args, problem, numba_transfer, backend: str):
    evaluator = TransferEvaluator(problem, numba_transfer, backend)
    lower = np.array([0.0, float(problem.min_tof)], dtype=np.float64)
    upper = np.array([float(args.opt_max_wait), float(args.opt_max_tof)], dtype=np.float64)
    x0 = np.array([0.5 * upper[0], 0.5 * (lower[1] + upper[1])], dtype=np.float64)
    bounds = Bounds(lower, upper)
    rng = np.random.default_rng(int(args.seed))
    es = cmaescpp.ACMA_C(
        2,
        bounds,
        x0=x0,
        input_sigma=float(args.opt_sigma),
        popsize=int(args.opt_popsize),
        max_evaluations=int(args.optimizer_evals),
        rg=rng,
    )

    best_seen = None
    best_feasible = None
    evals = 0
    stop = 0
    t0 = time.perf_counter()

    def record(info):
        nonlocal best_seen, best_feasible
        best_seen, best_feasible = _record_best(
            info,
            best_seen=best_seen,
            best_feasible=best_feasible,
            dv_tol=float(args.opt_dv_tol),
        )

    def evaluate(xs):
        nonlocal best_seen, best_feasible, evals
        x_arr = np.asarray(xs, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        evals += int(x_arr.shape[0])

        if backend == "numba":
            values, infos = numba_transfer.fixed_leg_objective_batch(
                int(args.opt_from),
                int(args.opt_to),
                float(args.opt_current_time),
                x_arr,
                min_tof=float(problem.min_tof),
                max_wait=float(args.opt_max_wait),
                max_tof=float(args.opt_max_tof),
                dv_limit=float(args.opt_dv_limit),
                penalty_linear=float(args.opt_penalty_linear),
                penalty_quad=float(args.opt_penalty_quad),
            )
            for row in infos:
                record(
                    {
                        "wait": float(row[0]),
                        "tof": float(row[1]),
                        "departure": float(row[2]),
                        "arrival": float(row[3]),
                        "dv": float(row[4]),
                        "excess": float(row[5]),
                    }
                )
            return values

        values = np.full(x_arr.shape[0], 1.0e30, dtype=np.float64)
        transfers = []
        valid = []
        infos = []
        for row_idx, x in enumerate(x_arr):
            wait = float(x[0])
            tof = float(x[1])
            dep = float(args.opt_current_time + wait)
            arr = float(dep + tof)
            info = {
                "wait": wait,
                "tof": tof,
                "departure": dep,
                "arrival": arr,
                "dv": float("inf"),
                "excess": float("inf"),
            }
            infos.append(info)
            if wait < lower[0] - 1e-12 or wait > upper[0] + 1e-12:
                continue
            if tof < lower[1] - 1e-12 or tof > upper[1] + 1e-12:
                continue
            valid.append(row_idx)
            transfers.append((int(args.opt_from), int(args.opt_to), dep, tof))

        if transfers:
            dvs = evaluator.compute_batch(transfers)
            for row_idx, dv in zip(valid, dvs):
                info = infos[row_idx]
                dv = float(dv)
                if not math.isfinite(dv):
                    continue
                excess = max(0.0, dv - float(args.opt_dv_limit))
                info["dv"] = dv
                info["excess"] = excess
                values[row_idx] = (
                    info["arrival"]
                    + float(args.opt_penalty_linear) * excess
                    + float(args.opt_penalty_quad) * excess * excess
                )
                record(info)
        return values

    evaluate(np.vstack((x0, lower, upper)))
    while evals < int(args.optimizer_evals):
        xs = es.ask()
        if xs is None:
            break
        ys = evaluate(xs)
        stop = int(es.tell(ys))
        if stop:
            break
    res = es.result()
    if res.x is not None:
        evaluate(np.asarray(res.x, dtype=np.float64))

    seconds = time.perf_counter() - t0
    chosen = best_feasible if best_feasible is not None else best_seen
    return {
        "backend": backend,
        "optimizer": "cma_ask_tell",
        "seconds": float(seconds),
        "evals": int(evals),
        "evals_per_second": float(evals / seconds),
        "feasible": best_feasible is not None,
        "best": chosen,
        "stop": int(stop),
        "result_fun": float(res.fun),
        "result_success": bool(res.success),
        "result_status": int(res.status),
    }


def _optimize_leg(args, problem, numba_transfer, backend: str):
    """BiteOpt optimizer path using ``minimize(fun, bounds)``.

    This avoids the ask/tell population loop. The Numba backend uses a scalar
    compiled fixed-leg objective, so each BiteOpt evaluation crosses Python only
    once and does the transfer plus penalty inside Numba.
    """
    lower = np.array([0.0, float(problem.min_tof)], dtype=np.float64)
    upper = np.array([float(args.opt_max_wait), float(args.opt_max_tof)], dtype=np.float64)
    x0 = np.array([0.5 * upper[0], 0.5 * (lower[1] + upper[1])], dtype=np.float64)
    bounds = Bounds(lower, upper)
    rng = np.random.default_rng(int(args.seed))

    best_seen = None
    best_feasible = None
    evals = 0

    def record(info):
        nonlocal best_seen, best_feasible
        best_seen, best_feasible = _record_best(
            info,
            best_seen=best_seen,
            best_feasible=best_feasible,
            dv_tol=float(args.opt_dv_tol),
        )

    def fitness(x):
        nonlocal evals
        evals += 1
        wait = float(x[0])
        tof = float(x[1])
        if backend == "numba":
            value, wait, tof, dep, arr, dv, excess = numba_transfer.fixed_leg_objective_one(
                int(args.opt_from),
                int(args.opt_to),
                float(args.opt_current_time),
                wait,
                tof,
                min_tof=float(problem.min_tof),
                max_wait=float(args.opt_max_wait),
                max_tof=float(args.opt_max_tof),
                dv_limit=float(args.opt_dv_limit),
                penalty_linear=float(args.opt_penalty_linear),
                penalty_quad=float(args.opt_penalty_quad),
            )
            info = {
                "wait": float(wait),
                "tof": float(tof),
                "departure": float(dep),
                "arrival": float(arr),
                "dv": float(dv),
                "excess": float(excess),
            }
            record(info)
            return float(value)

        dep = float(args.opt_current_time + wait)
        arr = float(dep + tof)
        info = {
            "wait": wait,
            "tof": tof,
            "departure": dep,
            "arrival": arr,
            "dv": float("inf"),
            "excess": float("inf"),
        }
        if wait < lower[0] - 1e-12 or wait > upper[0] + 1e-12:
            return 1.0e30
        if tof < lower[1] - 1e-12 or tof > upper[1] + 1e-12:
            return 1.0e30
        dv = float(problem.compute_transfer(int(args.opt_from), int(args.opt_to), dep, tof))
        if not math.isfinite(dv):
            return 1.0e30
        excess = max(0.0, dv - float(args.opt_dv_limit))
        info["dv"] = dv
        info["excess"] = excess
        record(info)
        return float(arr + float(args.opt_penalty_linear) * excess + float(args.opt_penalty_quad) * excess * excess)

    t0 = time.perf_counter()
    res = bitecpp.minimize(
        fitness,
        bounds,
        x0=x0,
        max_evaluations=int(args.optimizer_evals),
        M=int(args.bite_m),
        popsize=int(args.bite_popsize),
        stall_criterion=int(args.bite_stall_criterion),
        rg=rng,
    )
    seconds = time.perf_counter() - t0
    chosen = best_feasible if best_feasible is not None else best_seen
    nfev = int(getattr(res, "nfev", evals))
    return {
        "backend": backend,
        "optimizer": "bite_minimize",
        "seconds": float(seconds),
        "evals": nfev,
        "evals_per_second": float(nfev / seconds),
        "feasible": best_feasible is not None,
        "best": chosen,
        "stop": int(getattr(res, "status", 0)),
        "result_fun": float(res.fun),
        "result_success": bool(getattr(res, "success", False)),
        "result_status": int(getattr(res, "status", 0)),
    }


def _run_optimization(args, problem, numba_transfer):
    old, numba_threads = _set_threads(args.workers)
    try:
        warm_xs = np.column_stack((
            np.zeros(max(1, int(args.opt_popsize)), dtype=np.float64),
            np.full(max(1, int(args.opt_popsize)), max(float(problem.min_tof), 1.0), dtype=np.float64),
        ))
        numba_transfer.fixed_leg_objective_batch(
            int(args.opt_from),
            int(args.opt_to),
            float(args.opt_current_time),
            warm_xs,
            min_tof=float(problem.min_tof),
            max_wait=float(args.opt_max_wait),
            max_tof=float(args.opt_max_tof),
            dv_limit=float(args.opt_dv_limit),
            penalty_linear=float(args.opt_penalty_linear),
            penalty_quad=float(args.opt_penalty_quad),
        )
        numba_transfer.fixed_leg_objective_one(
            int(args.opt_from),
            int(args.opt_to),
            float(args.opt_current_time),
            0.0,
            max(float(problem.min_tof), 1.0),
            min_tof=float(problem.min_tof),
            max_wait=float(args.opt_max_wait),
            max_tof=float(args.opt_max_tof),
            dv_limit=float(args.opt_dv_limit),
            penalty_linear=float(args.opt_penalty_linear),
            penalty_quad=float(args.opt_penalty_quad),
        )
        pykep_cma = _optimize_leg_ask_tell(args, problem, numba_transfer, "pykep")
        numba_cma = _optimize_leg_ask_tell(args, problem, numba_transfer, "numba")
        pykep_bite = _optimize_leg(args, problem, numba_transfer, "pykep")
        numba_bite = _optimize_leg(args, problem, numba_transfer, "numba")
    finally:
        set_num_threads(old)
    return {
        "numba_threads": numba_threads,
        "settings": {
            "from": int(args.opt_from),
            "to": int(args.opt_to),
            "current_time": float(args.opt_current_time),
            "max_wait": float(args.opt_max_wait),
            "max_tof": float(args.opt_max_tof),
            "dv_limit": float(args.opt_dv_limit),
            "popsize": int(args.opt_popsize),
            "bite_popsize": int(args.bite_popsize),
            "bite_m": int(args.bite_m),
            "max_evals": int(args.optimizer_evals),
        },
        "results": [pykep_cma, numba_cma, pykep_bite, numba_bite],
    }


def _print_summary(payload):
    def print_rows(title, rows):
        print(f"\n{title}")
        print("| mode | samples/evals | seconds | calls/s | speedup |")
        print("|---|---:|---:|---:|---:|")
        for row in rows:
            print(
                f"| {row['mode']} | {row['samples']} | {row['seconds']:.6f} | "
                f"{row['calls_per_second']:.1f} | {row['speedup_vs_pykep_serial']:.2f}x |"
            )

    print_rows("Lambert API benchmark", payload["lambert"]["results"])
    print("Lambert correctness:", json.dumps(payload["lambert"]["correctness"], default=_json_default))
    print_rows("compute_transfer benchmark", payload["compute_transfer"]["results"])
    print("compute_transfer correctness:", json.dumps(payload["compute_transfer"]["correctness"], default=_json_default))

    print("\nLambert table creation benchmark")
    show_table_speedup = any("speedup_vs_pykep_table_batch" in row for row in payload["table_creation"]["results"])
    if show_table_speedup:
        print("| mode | evaluated cells | seconds | cells/s | pairs | nonempty pairs | retained cells | speedup vs pykep |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
    else:
        print("| mode | evaluated cells | seconds | cells/s | pairs | nonempty pairs | retained cells |")
        print("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["table_creation"]["results"]:
        base = (
            f"| {row['mode']} | {row['samples']} | {row['seconds']:.6f} | "
            f"{row['calls_per_second']:.1f} | {row['pair_count']} | "
            f"{row['nonempty_pairs']} | {row['retained_cells']} |"
        )
        if show_table_speedup:
            speedup = row.get("speedup_vs_pykep_table_batch")
            speedup_text = "" if speedup is None else f" {float(speedup):.2f}x"
            print(f"{base}{speedup_text} |")
        else:
            print(base)

    print("\nOptimizer demo")
    print("| optimizer | backend | evals | seconds | evals/s | feasible | arrival | dv | wait | tof |")
    print("|---|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for row in payload["optimization"]["results"]:
        best = row["best"] or {}
        print(
            f"| {row.get('optimizer', '')} | {row['backend']} | {row['evals']} | {row['seconds']:.6f} | "
            f"{row['evals_per_second']:.1f} | {row['feasible']} | "
            f"{float(best.get('arrival', float('nan'))):.9f} | "
            f"{float(best.get('dv', float('nan'))):.6f} | "
            f"{float(best.get('wait', float('nan'))):.9f} | "
            f"{float(best.get('tof', float('nan'))):.9f} |"
        )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", default=DEFAULT_PROBLEM, type=Path)
    ap.add_argument("--max-revs", type=int, nargs="+", default=[5, 2, 0])
    ap.add_argument("--samples", type=int, default=32768)
    ap.add_argument("--tof-max", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--optimizer-evals", type=int, default=4096)
    ap.add_argument("--opt-from", type=int, default=0)
    ap.add_argument("--opt-to", type=int, default=1)
    ap.add_argument("--opt-current-time", type=float, default=0.0)
    ap.add_argument("--opt-max-wait", type=float, default=12.0)
    ap.add_argument("--opt-max-tof", type=float, default=12.0)
    ap.add_argument("--opt-dv-limit", type=float, default=600.0)
    ap.add_argument("--opt-dv-tol", type=float, default=1.0e-7)
    ap.add_argument("--opt-penalty-linear", type=float, default=1000.0)
    ap.add_argument("--opt-penalty-quad", type=float, default=1.0)
    ap.add_argument("--opt-popsize", type=int, default=32)
    ap.add_argument("--opt-sigma", type=float, default=0.25)
    ap.add_argument("--bite-popsize", type=int, default=0)
    ap.add_argument("--bite-m", type=int, default=1)
    ap.add_argument("--bite-stall-criterion", type=int, default=0)
    ap.add_argument("--table-workers", type=int, default=16)
    ap.add_argument("--table-cluster", choices=("all", "inner", "outer", "switch"), default="all")
    ap.add_argument("--table-target-max-time", type=float, default=None)
    ap.add_argument("--table-min-tof", type=float, nargs="+", default=[0.05, 0.2, 0.35])
    ap.add_argument("--table-max-tof", type=float, nargs="+", default=[1.5, 5.0, 0.65])
    ap.add_argument("--table-tof-step", type=float, nargs="+", default=[0.05, 0.2, 0.1])
    ap.add_argument("--table-t-start-step", type=float, nargs="+", default=[0.05, 0.2, 0.1])
    ap.add_argument("--table-dv-limit", type=float, nargs="+", default=[105.0, 105.0, 605.0])
    ap.add_argument("--table-store-mode", choices=("best-per-departure", "all"), default="best-per-departure")
    ap.add_argument("--table-batch-chunk", type=int, default=1_000_000)
    ap.add_argument("--include-pykep-table", action="store_true", help="Also run the slow pykep table-creation comparison.")
    ap.add_argument("--table-pykep-workers", type=int, default=16)
    ap.add_argument("--table-pykep-chunksize", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    max_revs = _max_revs_arg(args.max_revs)
    problem = TomatoProblem(str(args.problem), max_revs=max_revs)
    numba_transfer = NumbaTransfer(problem)
    rows = _generate_transfer_rows(problem, int(args.samples), float(args.tof_max), int(args.seed))
    lambert_arrays = _prepare_lambert_cases(problem, numba_transfer, rows)
    _warm_numba(numba_transfer, rows, lambert_arrays)

    lambert = _benchmark_lambert(args, problem, numba_transfer, lambert_arrays)
    compute_transfer = _benchmark_compute_transfer(args, problem, numba_transfer, rows)
    table_creation = _benchmark_table_creation(args, problem, numba_transfer)
    optimization = _run_optimization(args, problem, numba_transfer)

    payload = {
        "meta": {
            "problem": _display_path(args.problem),
            "max_revs": max_revs,
            "samples": int(args.samples),
            "workers_requested": int(args.workers),
            "table_workers_requested": int(args.table_workers),
            "seed": int(args.seed),
            "optimizer_evals": int(args.optimizer_evals),
            "optimizer_popsize": int(args.opt_popsize),
            "python": sys.version.split()[0],
            "pykep": getattr(pk, "__version__", "unknown"),
        },
        "lambert": lambert,
        "compute_transfer": compute_transfer,
        "table_creation": table_creation,
        "optimization": optimization,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    _print_summary(payload)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
