"""資料模型與生成器"""
import numpy as np

from footballsim.data.generator import generate_league
from footballsim.data.repository import load_league, save_league
from footballsim.data.schema import GROUP_MODELS
from footballsim.model.attributes import ATTR_NAMES, GROUPS, INDEX, N_ATTR, compile_composites
from footballsim.model.positions import GROUP


def test_schema_matches_attribute_definitions():
    assert N_ATTR == 60
    for g, names in GROUPS.items():
        assert tuple(GROUP_MODELS[g].model_fields) == names


def test_generator_is_deterministic(tmp_path):
    a = generate_league(seed=5, n_teams=4)
    b = generate_league(seed=5, n_teams=4)
    assert a.model_dump_json() == b.model_dump_json()
    path = save_league(a, tmp_path / "lg.json")
    assert load_league(path).model_dump_json() == a.model_dump_json()


def test_positional_profiles_are_realistic():
    lg = generate_league(seed=2, n_teams=10)
    by = {}
    for t in lg.teams:
        for p in t.players:
            by.setdefault(GROUP[p.position], []).append((p.attr_vector(), p.height_cm))

    def mean(group, attr):
        return np.mean([v[INDEX[attr]] for v, _ in by[group]])

    assert mean("ST", "finishing") > mean("CB", "finishing") + 20
    assert mean("CB", "heading") > mean("W", "heading") + 10
    assert mean("W", "sprint_speed") > mean("CB", "sprint_speed") + 8
    assert mean("GK", "gk_reflexes") > mean("ST", "gk_reflexes") + 30
    assert np.mean([h for _, h in by["CB"]]) > np.mean([h for _, h in by["W"]]) + 8
    v = by["CM"][0][0]
    assert v.shape == (N_ATTR,) and v.min() >= 1 and v.max() <= 100


def test_composites_compile_and_normalise():
    w, C = compile_composites({"a": {"finishing": 2, "composure": 2}, "b": {"tackling": 1}})
    assert w.shape == (N_ATTR, 2)
    assert np.allclose(w.sum(0), 1.0)
    assert w[INDEX["finishing"], C.A] == 0.5
    assert set(ATTR_NAMES) == set(INDEX)
