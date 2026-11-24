"""核心公式：設計文件中標示「✔ 已驗證」的數字，以及單調性（性質測試）"""
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from footballsim.engine import core_math as cm

attr = st.floats(min_value=1, max_value=100)


def test_sprint_times_match_design_doc():
    v, t = cm.vmax_ms(60), cm.tau_s(60)
    assert cm.sprint_time(10, v, t) == pytest.approx(1.81, abs=0.01)
    assert cm.sprint_time(30, v, t) == pytest.approx(4.12, abs=0.01)
    assert cm.sprint_time(0, v, t) == 0.0


def test_fast_beats_explosive_only_over_distance():
    fast = (cm.vmax_ms(90), cm.tau_s(50))
    expl = (cm.vmax_ms(50), cm.tau_s(90))
    assert cm.sprint_time(5, *fast) > cm.sprint_time(5, *expl)
    assert cm.sprint_time(40, *fast) < cm.sprint_time(40, *expl)


def test_aerial_height_matters():
    def p(h, j):
        return dict(height=h, jumping=j, anticipation=65, heading=65, strength=65, bravery=65)
    assert cm.aerial_prob(p(175, 90), p(192, 60)) == pytest.approx(0.36, abs=0.01)
    assert cm.aerial_prob(p(185, 65), p(185, 65)) == pytest.approx(0.5)


def test_duel_examples():
    assert cm.duel_prob(0.47, 85, 65, 0.40, 0.40) == pytest.approx(0.664, abs=0.001)
    assert cm.duel_prob(0.86, 45, 65, 0.35, 0.15) == pytest.approx(0.753, abs=0.001)
    assert cm.duel_prob(0.86, 65, 65, 0.35, 0.15) == pytest.approx(0.86)


def test_shot_model_reproduces_xg_at_reference_and_real_sot_ratio():
    r = cm.shot_outcome_probs(0.10, 65, 65)
    assert r["goal"] == pytest.approx(0.10)
    assert r["on_target"] == pytest.approx(0.348, abs=0.001)   # 實測五大聯賽 34.9%
    assert cm.shot_outcome_probs(0.10, 85, 65)["goal"] / 0.10 == pytest.approx(1.23, abs=0.01)
    assert cm.shot_outcome_probs(0.10, 65, 85)["save_rate"] == pytest.approx(0.787, abs=0.001)


def test_base_xg_values():
    assert cm.base_xg(11, 0) == pytest.approx(0.167, abs=0.001)
    assert cm.base_xg(16.5, 0) == pytest.approx(0.082, abs=0.001)
    assert cm.base_xg(6, 0, header=True) == pytest.approx(0.198, abs=0.001)


def test_decisions_choose_best_more_often():
    u = [0.05, 0.03, 0.01, 0.0, -0.02]
    assert cm.choice_probs(u, 90, 65, 0)[0] == pytest.approx(0.80, abs=0.01)
    assert cm.choice_probs(u, 40, 65, 0)[0] == pytest.approx(0.58, abs=0.01)


def test_fatigue_factors():
    assert cm.energy_drain_per_min(50, 50) == pytest.approx(0.004)
    assert cm.physical_factor(0.9) == 1.0
    assert cm.physical_factor(0.55) == pytest.approx(0.925)
    assert cm.mental_factor(0.3, 90) > cm.mental_factor(0.3, 30)


@given(a=attr, b=attr, d=attr)
def test_duel_monotone_in_attacker_and_defender(a, b, d):
    lo, hi = min(a, b), max(a, b)
    assert cm.duel_prob(0.5, hi, d) >= cm.duel_prob(0.5, lo, d)
    assert cm.duel_prob(0.5, d, hi) <= cm.duel_prob(0.5, d, lo)


@given(s1=attr, s2=attr, g=attr, xg=st.floats(min_value=0.01, max_value=0.6))
def test_shot_goal_prob_monotone_and_bounded(s1, s2, g, xg):
    lo, hi = min(s1, s2), max(s1, s2)
    p_lo = cm.shot_outcome_probs(xg, lo, g)["goal"]
    p_hi = cm.shot_outcome_probs(xg, hi, g)["goal"]
    assert 0.0 <= p_lo <= p_hi <= 1.0
    assert cm.shot_outcome_probs(xg, hi, min(g + 10, 100))["goal"] <= p_hi + 1e-12


def test_shot_geometry_angle():
    dist, ang = cm.shot_geometry(11, 0)
    assert dist == 11
    assert ang == pytest.approx(2 * math.atan(3.66 / 11))
