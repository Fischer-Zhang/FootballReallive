"""M4：資料擬合工具與資料驅動的 xT"""
import numpy as np
import pandas as pd
import pytest

from footballsim.calibration import fit
from footballsim.config import load_params
from footballsim.engine.pitch import threat_for


def test_logistic_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(20000), rng.normal(0, 1, 20000)])
    y = (rng.random(20000) < 1 / (1 + np.exp(-(X @ [-1.0, 0.8])))).astype(float)
    w = fit.logistic_fit(X, y)
    assert w == pytest.approx([-1.0, 0.8], abs=0.06)


def test_xt_iteration_on_toy_data():
    # 兩個格子：從中場傳到門前（成功），門前射門 xG 0.3 → 中場 xT ≈ 0.3 × 移動機率
    rows = [dict(type="pass", sub="Open Play", x=50, y=34, ex=100, ey=34, ok=1, xg=np.nan)] * 10
    rows += [dict(type="shot", sub="Open Play", x=100, y=34, ex=np.nan, ey=np.nan, ok=0, xg=0.3)] * 10
    grid, info = fit.expected_threat(pd.DataFrame(rows))
    goal_cell = grid[int(100 / 105 * 16), int(34 / 68 * 12)]
    mid_cell = grid[int(50 / 105 * 16), int(34 / 68 * 12)]
    assert goal_cell == pytest.approx(0.3)
    assert mid_cell == pytest.approx(0.3)
    assert info["shots"] == 10


def test_engine_uses_statsbomb_xt_grid():
    p = load_params()
    assert p.threat.source == "grid"
    t = threat_for(p)
    central = [t.xt_at(x, 34) for x in (20, 50, 75, 90, 100)]
    assert central == sorted(central)                      # 越接近對方球門越有威脅
    assert t.xt_at(100, 34) > 5 * t.xt_at(75, 34)          # 門前遠高於禁區外
    assert t.xt_at(95, 34) > t.xt_at(95, 6)                # 中路高於邊路


def test_fitted_xg_headers_much_worse_than_feet():
    t = threat_for(load_params())
    assert t.xg_at(99, 34, header=True) < 0.5 * t.xg_at(99, 34)
