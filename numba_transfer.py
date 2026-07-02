"""Drop-in compute_transfer wrappers backed by the local Numba Lambert port."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

try:  # package import
    from . import pk_lambert as pl
except ImportError:  # script import from this directory
    import pk_lambert as pl


DAY = 86400.0


class NumbaTransfer:
    """Vectorized replacement for ``TomatoProblem.compute_transfer``.

    The scalar method has the same call signature as the pykep-backed original:
    ``compute_transfer(i_from, i_to, t_start_days, tof_days) -> dv_m_per_s``.
    ``compute_transfer_batch`` evaluates rows of the same four-tuple using a
    Numba ``prange`` kernel.
    """

    def __init__(self, problem):
        el = np.asarray(problem.tomato_orbital_parameters, dtype=np.float64)
        self.A = np.ascontiguousarray(el[:, 0])
        self.E = np.ascontiguousarray(el[:, 1])
        self.I = np.ascontiguousarray(el[:, 2])
        self.O = np.ascontiguousarray(el[:, 3])
        self.Oc = np.ascontiguousarray(el[:, 4])
        self.M0 = np.ascontiguousarray(el[:, 5])
        self.mu = float(problem.mu_moon)
        self.NN = np.sqrt(self.mu / self.A**3)
        self._problem = problem
        mr = np.asarray(problem.max_revs, dtype=np.int32).reshape(-1)
        self._max_revs_value = mr
        self._scalar_max_revs = mr.size == 1

        n = len(self.A)
        self.max_revs_pair = np.empty((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                self.max_revs_pair[i, j] = self._max_revs(i, j)

    def _max_revs(self, i_from: int, i_to: int) -> int:
        if self._scalar_max_revs:
            max_revs = int(self._max_revs_value[0])
            if self._problem.is_cluster_switch(int(i_from), int(i_to)):
                max_revs += 1
            return max_revs
        transfer_type = int(self._problem.transfer_type(int(i_from), int(i_to)))
        return int(self._max_revs_value[transfer_type])

    def compute_transfer(self, i_from: int, i_to: int, t_start: float, tof: float) -> float:
        i = int(i_from)
        j = int(i_to)
        return float(
            pl.dv_full(
                self.A[i],
                self.E[i],
                self.I[i],
                self.O[i],
                self.Oc[i],
                self.M0[i],
                self.NN[i],
                self.A[j],
                self.E[j],
                self.I[j],
                self.O[j],
                self.Oc[j],
                self.M0[j],
                self.NN[j],
                self.mu,
                float(t_start) * DAY,
                float(tof) * DAY,
                self._max_revs(i, j),
            )
        )

    def compute_transfer_batch(self, transfers: Iterable[tuple[int, int, float, float]]) -> list[float]:
        rows = list(transfers)
        if not rows:
            return []
        arr = np.asarray(rows, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("transfers must be rows of (i_from, i_to, t_start, tof)")

        i_idx = np.ascontiguousarray(arr[:, 0].astype(np.int64))
        j_idx = np.ascontiguousarray(arr[:, 1].astype(np.int64))
        td_sec = np.ascontiguousarray(arr[:, 2] * DAY)
        tof_sec = np.ascontiguousarray(arr[:, 3] * DAY)
        out = np.empty(arr.shape[0], dtype=np.float64)
        pl.dv_full_batch(
            self.A,
            self.E,
            self.I,
            self.O,
            self.Oc,
            self.M0,
            self.NN,
            self.mu,
            i_idx,
            j_idx,
            td_sec,
            tof_sec,
            self.max_revs_pair,
            out,
        )
        return out.tolist()

    def fixed_leg_objective_batch(
        self,
        i_from: int,
        i_to: int,
        current_time: float,
        xs,
        *,
        min_tof: float,
        max_wait: float,
        max_tof: float,
        dv_limit: float,
        penalty_linear: float,
        penalty_quad: float,
    ):
        """Evaluate ``[wait, tof]`` population rows for one leg in compiled code.

        Returns ``(values, infos)`` where ``infos`` columns are wait, tof,
        departure, arrival, dv, and excess.
        """
        x_arr = np.asarray(xs, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2 or x_arr.shape[1] != 2:
            raise ValueError("xs must be a batch of [wait, tof] rows")
        x_arr = np.ascontiguousarray(x_arr)
        values = np.empty(x_arr.shape[0], dtype=np.float64)
        infos = np.empty((x_arr.shape[0], 6), dtype=np.float64)
        i = int(i_from)
        j = int(i_to)
        pl.fixed_leg_objective_batch(
            self.A,
            self.E,
            self.I,
            self.O,
            self.Oc,
            self.M0,
            self.NN,
            self.mu,
            i,
            j,
            float(current_time) * DAY,
            x_arr,
            float(min_tof) * DAY,
            float(max_wait) * DAY,
            float(max_tof) * DAY,
            float(dv_limit),
            float(penalty_linear),
            float(penalty_quad),
            int(self.max_revs_pair[i, j]),
            values,
            infos,
        )
        return values, infos

    def fixed_leg_objective_one(
        self,
        i_from: int,
        i_to: int,
        current_time: float,
        wait: float,
        tof: float,
        *,
        min_tof: float,
        max_wait: float,
        max_tof: float,
        dv_limit: float,
        penalty_linear: float,
        penalty_quad: float,
    ):
        """Evaluate one ``[wait, tof]`` point for one leg in compiled code."""
        i = int(i_from)
        j = int(i_to)
        return pl.fixed_leg_objective_one(
            self.A,
            self.E,
            self.I,
            self.O,
            self.Oc,
            self.M0,
            self.NN,
            self.mu,
            i,
            j,
            float(current_time) * DAY,
            float(wait),
            float(tof),
            float(min_tof) * DAY,
            float(max_wait) * DAY,
            float(max_tof) * DAY,
            float(dv_limit),
            float(penalty_linear),
            float(penalty_quad),
            int(self.max_revs_pair[i, j]),
        )


def make_compute_transfer(problem, legacy: bool | None = None):
    """Return pykep or Numba scalar ``compute_transfer``."""
    if legacy is None:
        legacy = os.environ.get("CL_LEGACY_LAMBERT", "0") not in ("", "0", "false", "False")
    if legacy:
        return problem.compute_transfer
    return NumbaTransfer(problem).compute_transfer


def make_compute_transfer_batch(problem, legacy: bool | None = None):
    """Return pykep-loop or Numba-parallel batched ``compute_transfer``."""
    if legacy is None:
        legacy = os.environ.get("CL_LEGACY_LAMBERT", "0") not in ("", "0", "false", "False")
    if legacy:
        def _legacy_batch(transfers):
            return [
                float(problem.compute_transfer(int(i), int(j), float(t_start), float(tof)))
                for i, j, t_start, tof in transfers
            ]

        return _legacy_batch
    return NumbaTransfer(problem).compute_transfer_batch
