"""KTTSP transfer evaluation backed by the published ``pykep-rust`` wheel."""
from __future__ import annotations

import math
from importlib.metadata import version
from typing import Iterable

import numpy as np
import pykep_rust as pk


DAY = 86_400.0
PYKEP_RUST_VERSION = version("pykep-rust")


def _difference_norm(left, right) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def pykep_rust_lambert_dv(case) -> float:
    """Return the best Lambert delta-v for one pre-propagated benchmark case."""
    r1, v_dep, r2, v_arr, tof_sec, mu, max_revs = case
    best_dv = math.inf
    for clockwise in (False, True):
        try:
            problem = pk.LambertProblem(
                r1,
                r2,
                float(tof_sec),
                float(mu),
                clockwise,
                int(max_revs),
            )
        except ValueError:
            continue
        for solution in problem.solutions:
            delta_v = _difference_norm(solution.departure_velocity, v_dep)
            delta_v += _difference_norm(solution.arrival_velocity, v_arr)
            best_dv = min(best_dv, delta_v)
    return best_dv


class PykepRustTransfer:
    """Scalar ``compute_transfer`` implementation using the PyPI native wheel."""

    def __init__(self, problem):
        self._problem = problem
        self.mu = float(problem.mu_moon)
        self._max_revs_value = np.asarray(problem.max_revs, dtype=np.int32).reshape(-1)
        self._scalar_max_revs = self._max_revs_value.size == 1
        reference_epoch = float(problem.t0.mjd2000)
        self.planets = []
        for index, source_planet in enumerate(problem.tomatoes):
            position, velocity = source_planet.eph(reference_epoch)
            state = [float(value) for value in (*position, *velocity)]
            self.planets.append(
                pk.Planet.keplerian_from_state(
                    reference_epoch,
                    state,
                    self.mu,
                    f"easy.kttsp body {index}",
                )
            )

    def _max_revs(self, i_from: int, i_to: int) -> int:
        if self._scalar_max_revs:
            max_revs = int(self._max_revs_value[0])
            if self._problem.is_cluster_switch(int(i_from), int(i_to)):
                max_revs += 1
            return max_revs
        transfer_type = int(self._problem.transfer_type(int(i_from), int(i_to)))
        return int(self._max_revs_value[transfer_type])

    def compute_transfer(
        self,
        i_from: int,
        i_to: int,
        t_start: float,
        tof: float,
    ) -> float:
        """Return the best two-impulse delta-v for one transfer, in m/s."""
        departure = self.planets[int(i_from)].state(float(t_start))
        arrival = self.planets[int(i_to)].state(float(t_start) + float(tof))
        return pykep_rust_lambert_dv(
            (
                departure[:3],
                departure[3:],
                arrival[:3],
                arrival[3:],
                float(tof) * DAY,
                self.mu,
                self._max_revs(int(i_from), int(i_to)),
            )
        )

    def compute_transfer_batch(
        self,
        transfers: Iterable[tuple[int, int, float, float]],
    ) -> list[float]:
        """Evaluate transfer rows serially through the native Python extension."""
        return [
            self.compute_transfer(int(i), int(j), float(t_start), float(tof))
            for i, j, t_start, tof in transfers
        ]
