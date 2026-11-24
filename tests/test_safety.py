"""低負荷與失控防護：批次預設單行程、數值函式庫單執行緒、異常的比賽會立即中止"""
import os

import pytest

import footballsim  # noqa: F401  （載入時設定單執行緒）
from footballsim.calibration import runner
from footballsim.data.generator import generate_league
from footballsim.engine import MatchSim
from footballsim.engine.state import Outcome


def test_low_load_defaults():
    assert runner.DEFAULT_WORKERS == 1 and runner.MAX_WORKERS <= 2
    assert runner.MEM_CAP_GB <= 4
    assert os.environ.get("OPENBLAS_NUM_THREADS") is not None


def test_match_rejects_non_advancing_outcome():
    lg = generate_league(seed=1, n_teams=2)
    sim = MatchSim(lg.teams[0], lg.teams[1], seed=0)
    with pytest.raises(RuntimeError):
        sim.apply(Outcome(duration=0.0, events=[], team=0, carrier=1, ball=(52.5, 34.0)))
    with pytest.raises(RuntimeError):
        sim.apply(Outcome(duration=float("nan"), events=[], team=0, carrier=1, ball=(52.5, 34.0)))
