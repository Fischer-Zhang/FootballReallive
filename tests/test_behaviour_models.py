"""行為模型與「不使用全域偏好補丁」的原則"""
import math

from footballsim.config import FITTED, FITTED360, load_params, load_raw
from footballsim.engine.core_math import shot_geometry, sigmoid

REMOVED_PATCHES = {"shot_bias", "cross_bias", "clear_bias", "mentality_shot", "mentality_cross",
                   "lead_directness"}


def test_no_global_preference_patches():
    """決策偏好只允許戰術指示與球員特質；不得用全域常數把統計「推」到目標值"""
    decision = load_raw()["decision"]
    assert not (REMOVED_PATCHES & set(decision)), REMOVED_PATCHES & set(decision)


def test_fitted_models_override_defaults():
    assert FITTED.exists() and FITTED360.exists()
    p = load_params()
    for section in ("shot_model", "transitions", "shot_propensity", "cross_propensity", "clear_propensity"):
        assert hasattr(p, section), section


def test_behaviour_models_use_geometry_not_annotation():
    """行為模型只用引擎也算得出的位置特徵；不得使用人工標記的「受壓迫」"""
    raw = load_raw()
    for sec, banned in (("shot_propensity", {"a_press", "press_dist"}), ("cross_propensity", {"c_press"})):
        assert not (banned & set(raw[sec])), (sec, banned & set(raw[sec]))


def _p_shoot(p, dist_m, d_near):
    sp = p.shot_propensity
    d, ang = shot_geometry(dist_m, 0.0)
    return sigmoid(sp.a0 + sp.a_dist * d + sp.a_logang * math.log(ang) + sp.a_near * d_near)


def test_shot_propensity_behaves_like_real_players():
    p = load_params()
    probs = [_p_shoot(p, d, 2.0) for d in (8, 12, 18, 25, 32)]
    assert probs == sorted(probs, reverse=True)            # 越遠越少射門
    assert probs[0] > 0.5 and probs[-1] < 0.1
    assert _p_shoot(p, 18, 0.8) < _p_shoot(p, 18, 5.0)     # 防守者貼身時較少射門
    assert p.shot_propensity.k_quality > 0                 # 封堵越多（射門品質越差）越少射門


def test_cross_and_clear_propensity_directions():
    p = load_params()
    cp, kp = p.cross_propensity, p.clear_propensity
    near_byline = sigmoid(cp.c0 + cp.c_depth * 5 + cp.c_wide * 25 + cp.c_near * 3)
    far = sigmoid(cp.c0 + cp.c_depth * 22 + cp.c_wide * 25 + cp.c_near * 3)
    assert near_byline > far                                # 越接近底線越常傳中
    assert sigmoid(kp.k0 + kp.k_x * 6 + kp.k_near * 3) > sigmoid(kp.k0 + kp.k_x * 30 + kp.k_near * 3)
    assert sigmoid(kp.k0 + kp.k_x * 15 + kp.k_near * 1) > sigmoid(kp.k0 + kp.k_x * 15 + kp.k_near * 6)
