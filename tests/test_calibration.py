"""校準工具：目標檔、彙整、比對、敏感度用的屬性加成（不跑大量模擬）"""
import yaml

from footballsim.calibration.runner import aggregate, compare, run_batch
from footballsim.calibration.sensitivity import boost
from footballsim.config import CONFIG_DIR
from footballsim.data.generator import generate_league
from footballsim.model.positions import GROUP


def test_targets_file_has_real_benchmarks():
    doc = yaml.safe_load((CONFIG_DIR / "targets" / "top5.yaml").read_text(encoding="utf-8"))
    t = doc["top5"]
    assert t["matches"] > 7000
    assert 2.6 < t["goals_pm"] < 3.0
    assert set(doc["top5_range"]) >= {"goals_pm", "shots_pt", "fouls_pt", "HT_leader_win%"}


def test_small_batch_aggregates_and_compares():
    lg = generate_league(seed=4, n_teams=4)
    S = run_batch(lg, 4, workers=1, seed=1)
    m = aggregate(S)
    assert m["matches"] == 4
    assert 0 < m["shots_pt"] < 40
    assert 60 < m["pass_pct"] < 95
    rows = compare(m)
    assert rows and all(r[5] in "✓~✗" for r in rows)


def test_boost_only_touches_selected_positions():
    lg = generate_league(seed=4, n_teams=3)
    out = boost(lg, 1, "finishing", ["ST"], 10)
    for before, after in zip(lg.teams[1].players, out.teams[1].players):
        diff = after.shooting.finishing - before.shooting.finishing
        if GROUP[before.position] == "ST":
            assert diff == min(10, 100 - before.shooting.finishing)
        else:
            assert diff == 0
    assert out.teams[0].model_dump() == lg.teams[0].model_dump()
