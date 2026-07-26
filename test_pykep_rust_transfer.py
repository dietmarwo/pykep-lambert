"""Installed-wheel integration tests for the pykep-rust benchmark backend."""

import pytest

from pykep_rust_transfer import PYKEP_RUST_VERSION, PykepRustTransfer
from utils_users import TomatoProblem


def test_published_wheel_matches_original_pykep() -> None:
    problem = TomatoProblem("problems/easy.kttsp", max_revs=[5, 2, 0])
    transfer = PykepRustTransfer(problem)
    cases = [
        (0, 1, 0.0, 1.0),
        (0, 1, 12.0, 12.0),
        (4, 0, 0.5, 0.8),
        (4, 11, 0.25, 0.8),
    ]
    assert {problem.transfer_type(case[0], case[1]) for case in cases} == {0, 1, 2}
    expected = [problem.compute_transfer(*case) for case in cases]
    actual = transfer.compute_transfer_batch(cases)
    assert PYKEP_RUST_VERSION == "0.1.1"
    assert actual == pytest.approx(expected, rel=2e-12, abs=5e-7)
