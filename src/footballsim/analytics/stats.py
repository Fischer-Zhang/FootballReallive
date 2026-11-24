"""由事件流計算球隊與球員統計（§7）"""
from __future__ import annotations

from collections import defaultdict

from ..engine import events as E

_CHAIN_BREAKERS = {E.INTERCEPTION, E.RECOVERY, E.KEEPER_SAVE, E.KEEPER_CLAIM}


def match_stats(res) -> dict:
    teams = [defaultdict(float), defaultdict(float)]
    players: dict = defaultdict(lambda: defaultdict(float))
    goals = []
    last_pass = {0: None, 1: None}

    for ev in res.events:
        t = ev.team
        if t not in (0, 1):
            continue
        T = teams[t]
        P = players[ev.player] if ev.player else None
        typ, r = ev.type, ev.result

        if typ in _CHAIN_BREAKERS or (typ == E.TACKLE and r == E.SUCCESS):
            last_pass[1 - t] = None
            if typ == E.RECOVERY:
                last_pass[t] = None

        if typ in E.SHOT_TYPES:
            T["shots"] += 1
            T["xg"] += ev.xg or 0.0
            if P is not None:
                P["shots"] += 1
                P["xg"] += ev.xg or 0.0
            if r in (E.GOAL, E.SAVED):
                T["sot"] += 1
                if P is not None:
                    P["sot"] += 1
            if r == E.BLOCKED:
                T["blocked_shots"] += 1
            if typ == E.SHOT_PENALTY:
                T["penalties"] += 1
            if typ == E.SHOT_FREEKICK:
                T["direct_fk"] += 1
            if ev.body == "head":
                T["header_shots"] += 1
            assister = last_pass[t] if typ != E.SHOT_PENALTY else None
            if assister and assister != ev.player:
                players[assister]["key_passes"] += 1
            if r == E.GOAL:
                T["goals"] += 1
                if P is not None:
                    P["goals"] += 1
                if ev.body == "head":
                    T["header_goals"] += 1
                if typ == E.SHOT_PENALTY:
                    T["penalty_goals"] += 1
                if assister and assister != ev.player:
                    players[assister]["assists"] += 1
                goals.append({"team": t, "player": ev.player, "minute": ev.minute, "type": typ,
                              "body": ev.body, "assist": assister if assister != ev.player else None,
                              "xg": ev.xg})
            last_pass[t] = None
        elif typ in E.PASS_LIKE:
            ok = r == E.SUCCESS
            if typ == E.THROW_IN:
                T["throw_ins"] += 1
            else:
                T["passes"] += 1
                T["passes_ok"] += ok
                if P is not None:
                    P["passes"] += 1
                    P["passes_ok"] += ok
            if typ == E.CROSS:
                T["crosses"] += 1
                T["crosses_ok"] += ok
            if typ == E.THROUGH:
                T["through_balls"] += 1
                T["through_ok"] += ok
            if typ == E.CORNER_SHORT:
                T["corners"] += 1
            if typ == E.GOAL_KICK:
                T["goal_kicks"] += 1
            if r == E.OFFSIDE:
                T["offsides"] += 1
            last_pass[t] = ev.player if ok else None
        elif typ in (E.CORNER, E.FREEKICK_CROSSED):
            if typ == E.CORNER:
                T["corners"] += 1
            T["set_piece_crosses"] += 1
            last_pass[t] = ev.player if r == E.SUCCESS else None
        elif typ == E.FOUL:
            T["fouls"] += 1
            if P is not None:
                P["fouls"] += 1
        elif typ == E.CARD:
            if r == E.YELLOW:
                T["yellow"] += 1
                if P is not None:
                    P["yellow"] += 1
            else:
                T["red"] += 1
                if r == E.SECOND_YELLOW:
                    T["yellow"] += 1
                if P is not None:
                    P["red"] += 1
        elif typ == E.TAKE_ON:
            T["take_ons"] += 1
            T["take_ons_ok"] += r == E.SUCCESS
            if P is not None:
                P["take_ons"] += 1
                P["take_ons_ok"] += r == E.SUCCESS
        elif typ == E.TACKLE:
            T["tackles"] += 1
            T["tackles_won"] += r == E.SUCCESS
            if P is not None:
                P["tackles_won"] += r == E.SUCCESS
        elif typ == E.INTERCEPTION:
            T["interceptions"] += 1
            if P is not None:
                P["interceptions"] += 1
        elif typ == E.CLEARANCE:
            T["clearances"] += 1
        elif typ == E.AERIAL:
            T["aerials_won"] += 1
            if P is not None:
                P["aerials_won"] += 1
        elif typ == E.KEEPER_SAVE:
            T["saves"] += 1
            if P is not None:
                P["saves"] += 1
        elif typ == E.KEEPER_CLAIM:
            T["claims"] += 1
        elif typ == E.DRIBBLE:
            T["carries"] += 1
        elif typ == E.BAD_TOUCH:
            T["bad_touches"] += 1
        elif typ == E.RECOVERY:
            T["recoveries"] += 1
        elif typ == E.SUBSTITUTION:
            T["subs"] += 1
        elif typ == E.INJURY:
            T["injuries"] += 1
            T["injuries_forced"] += r == "forced_off"

    poss = res.possession_time
    tot = sum(poss) or 1.0
    for t in (0, 1):
        teams[t]["possession"] = 100.0 * poss[t] / tot
        teams[t]["pass_pct"] = 100.0 * teams[t]["passes_ok"] / max(teams[t]["passes"], 1)
    for pid, P in players.items():
        P["rating"] = _rating(P, res.minutes.get(pid, 0.0))
    return {"teams": [dict(t) for t in teams], "players": {k: dict(v) for k, v in players.items()},
            "goals": goals}


def _rating(P: dict, minutes: float) -> float:
    """簡化評分（M5 改為 VAEP 動作價值）：6.0 為基準"""
    if minutes <= 0:
        return 0.0
    r = (6.0 + 1.0 * P.get("goals", 0) + 0.6 * P.get("assists", 0) + 0.12 * P.get("key_passes", 0)
         + 0.015 * P.get("passes_ok", 0) - 0.03 * (P.get("passes", 0) - P.get("passes_ok", 0))
         + 0.10 * P.get("take_ons_ok", 0) + 0.12 * P.get("tackles_won", 0)
         + 0.10 * P.get("interceptions", 0) + 0.06 * P.get("aerials_won", 0)
         + 0.20 * P.get("saves", 0) + 0.15 * P.get("sot", 0) - 0.3 * P.get("yellow", 0)
         - 1.5 * P.get("red", 0) - 0.08 * P.get("fouls", 0))
    return round(min(10.0, max(3.0, r)), 1)
