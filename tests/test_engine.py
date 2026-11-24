"""比賽引擎：可重現性與規則不變量（每個測試只跑少數幾場）"""
import pytest

from footballsim.data.generator import generate_league
from footballsim.engine import events as E
from footballsim.engine import simulate_match
from footballsim.engine.match import MatchSim


@pytest.fixture(scope="module")
def league():
    return generate_league(seed=3, n_teams=6)


@pytest.fixture(scope="module")
def result(league):
    return simulate_match(league.teams[0], league.teams[1], seed=11)


def _sig(res):
    return [(e.period, round(e.t, 6), e.team, e.player, e.type, e.result) for e in res.events]


def test_same_seed_same_match(league, result):
    again = simulate_match(league.teams[0], league.teams[1], seed=11)
    assert _sig(again) == _sig(result)
    other = simulate_match(league.teams[0], league.teams[1], seed=12)
    assert _sig(other) != _sig(result)


def test_score_matches_goal_events(result):
    goals = [0, 0]
    for e in result.events:
        if e.type in E.SHOT_TYPES and e.result == E.GOAL:
            goals[e.team] += 1
    assert tuple(goals) == result.score
    assert result.ht_score[0] <= result.score[0] and result.ht_score[1] <= result.score[1]


def test_clock_and_rules(result):
    assert 85 * 60 > result.ball_in_play > 40 * 60          # 淨比賽時間合理
    for half in (1, 2):
        ts = [e.t for e in result.events if e.period == half]
        assert ts == sorted(ts)
        assert max(ts) >= 45 * 60
    assert all(s <= 5 for s in result.subs)
    st = result.stats()["teams"]
    assert 99.0 < st[0]["possession"] + st[1]["possession"] < 101.0
    assert st[0]["passes"] > 150 and st[1]["passes"] > 150


def test_red_card_reduces_players(league):
    sim = MatchSim(league.teams[2], league.teams[3], seed=5)
    from footballsim.engine import referee
    referee.send_off(sim, 11 * 1 + 4)
    assert sim.active[11:].sum() == 10
    res = sim.run()
    assert res.score[0] >= 0


def test_all_formations_run(league):
    for i, f in enumerate(["4-3-3", "4-2-3-1", "4-4-2", "3-5-2"]):
        home = league.teams[4].model_copy(deep=True)
        home.tactics.formation = f
        res = simulate_match(home, league.teams[5], seed=i)
        assert sum(res.score) >= 0 and len(res.events) > 500


def test_duplicate_player_ids_are_rejected(league):
    clone = league.teams[0].model_copy(deep=True)
    with pytest.raises(ValueError):
        MatchSim(league.teams[0], clone, seed=1)
