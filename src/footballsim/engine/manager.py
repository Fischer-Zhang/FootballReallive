"""教練 AI：比分情境、換人、傷病處理（§4.15、§4.17）"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..data.schema import Player
from ..model.attributes import ATTR
from ..model.positions import DEFENSIVE_GROUPS, Pos, position_rating
from . import events as E


# ------------------------------------------------------------------ 每分鐘
def minute(sim) -> None:
    update_mentality(sim)
    _injury_hazard(sim)


def update_mentality(sim) -> None:
    """落後 → 心態提高；領先 → 收縮；少一人 → 保守"""
    gs = sim.p.game_state
    m = sim.match_minute()
    for t in (0, 1):
        me, op = sim.teams[t], sim.teams[1 - t]
        diff = me.score - op.score
        off = 0.0
        if diff < 0 and m >= gs.trail_start:
            off = min(gs.trail_max, (m - gs.trail_start) / gs.trail_rate) * (1.0 if diff == -1 else 0.7)
        elif diff > 0 and m >= gs.lead_start:
            # 領先後退守的程度依相對實力而定：弱隊領先會擺大巴，強隊領先多半繼續壓制
            edge = (me.strength - op.strength) / gs.strength_scale
            protect = min(max(1.0 - edge, gs.protect_min), gs.protect_max)
            off = -min(gs.lead_max, (m - gs.lead_start) / gs.lead_rate) * protect
        off += gs.red_card * (me.red_cards - op.red_cards)
        me.mentality_offset = off


def _injury_hazard(sim) -> None:
    ip, rng = sim.p.injury, sim.rng_misc
    for s in np.flatnonzero(sim.active):
        if sim.need_off[s]:
            continue
        prone = sim.base[s, ATTR.INJURY_PRONENESS]
        h = ip.per_min * (prone / 50) * (1 + 2 * max(0.0, 0.6 - sim.energy[s]))
        if rng.random() < h:
            _injure(sim, int(s))


def contact_injury(sim, s: int, evs: list, frame: int, x: float, y: float) -> None:
    """被犯規時的接觸傷"""
    ip = sim.p.injury
    prone = sim.base[s, ATTR.INJURY_PRONENESS]
    if sim.rng_misc.random() < ip.contact * (prone / 50):
        _injure(sim, s, evs, frame, x, y)


def _injure(sim, s: int, evs: Optional[list] = None, frame: int = 0, x: float = 52.5,
            y: float = 34.0) -> None:
    ip = sim.p.injury
    forced = sim.rng_misc.random() < ip.forced_off_share
    ev = sim.ev(frame, s, E.INJURY, "forced_off" if forced else "knock", x, y)
    (evs if evs is not None else sim.events).append(ev)
    sim.extra_dead[sim.period - 1] += sim.p.stoppage.injury
    if forced:
        sim.need_off[s] = True
        sim.knock[s] = max(sim.knock[s], 0.25)    # 換下前大幅受限
    else:
        sim.knock[s] = max(sim.knock[s], ip.knock)


# ------------------------------------------------------------------ 換人
def halftime(sim) -> None:
    for t in (0, 1):
        _consider(sim, t, 45.0, at_halftime=True)


def at_stoppage(sim) -> None:
    m = sim.match_minute()
    for t in (0, 1):
        _consider(sim, t, m, at_halftime=False)


def _bench_available(sim, t: int) -> list:
    used = set(sim.minutes_on)
    return [p for p in sim.teams[t].bench if p.player_id not in used]


def _best_replacement(sim, t: int, slot_pos: Pos) -> Optional[Player]:
    cands = _bench_available(sim, t)
    if slot_pos == Pos.GK:
        cands = [p for p in cands if p.position == Pos.GK]
    else:
        cands = [p for p in cands if p.position != Pos.GK]
    if not cands:
        return None
    return max(cands, key=lambda p: position_rating(sim.day_vec[p.player_id], slot_pos)
               * (0.55 + 0.45 * p.familiarity(slot_pos) / 100))


def _substitute(sim, t: int, s: int, new: Player, minute: float) -> None:
    old = sim.players[s]
    sim.minutes_off[old.player_id] = minute
    sim.events.append(sim.ev(0, s, E.SUBSTITUTION, "", 52.5, 34.0,
                             extra={"off": old.player_id, "on": new.player_id}))
    sim.put(s, new, minute)
    sim.teams[t].subs_used += 1
    sim.extra_dead[sim.period - 1] += sim.p.stoppage.sub


def _consider(sim, t: int, minute: float, at_halftime: bool) -> None:
    mg = sim.p.manager
    team = sim.teams[t]
    if team.subs_used >= mg.max_subs:
        return
    if not at_halftime and team.windows_used >= mg.max_windows:
        return
    act = sim.active_slots(t)
    urgent = [s for s in act if sim.need_off[s]]
    planned_target = 0
    planned = mg.planned_subs if isinstance(mg.planned_subs, dict) else vars(mg.planned_subs)
    for k, v in planned.items():
        if minute >= float(k):
            planned_target = max(planned_target, int(v))
    tired = []
    if minute >= 55 or at_halftime:
        for s in act:
            if sim.is_gk(s) or s in urgent:
                continue
            e = float(sim.energy[s])
            pri = 0.0
            if e < mg.energy_threshold:
                pri = 50 + 100 * (mg.energy_threshold - e)
            elif sim.yellow[s] and team.slot_group[s % 11] in DEFENSIVE_GROUPS \
                    and sim.eff[s, ATTR.AGGRESSION] > 65 and minute >= 60:
                pri = 20
            tired.append((pri, -e, s))
    tired.sort(reverse=True)
    n_urgent_tired = sum(1 for pri, _, _ in tired if pri >= 50)
    n_planned = max(0, planned_target - team.subs_used)
    n_now = len(urgent) + max(n_urgent_tired, n_planned)
    if at_halftime:
        n_now = len(urgent) + n_urgent_tired
    n_now = min(n_now, mg.max_subs - team.subs_used, 3)
    if n_now <= 0:
        return
    order = urgent + [s for _, _, s in tired]
    made = 0
    for s in order:
        if made >= n_now:
            break
        new = _best_replacement(sim, t, team.slot_pos[s % 11])
        if new is None:
            continue
        _substitute(sim, t, s, new, minute)
        made += 1
    if made and not at_halftime:
        team.windows_used += 1


def replace_goalkeeper(sim, t: int) -> None:
    """門將被罰下：有換人名額就換上替補門將並撤下一名外野球員；否則由外野球員頂替"""
    team = sim.teams[t]
    gk_slot = 11 * t
    out = [s for s in sim.active_slots(t) if s != gk_slot]
    if not out:
        return
    weakest = min(out, key=lambda s: position_rating(sim.base[s], team.slot_pos[s % 11]))
    new = _best_replacement(sim, t, Pos.GK) if team.subs_used < sim.p.manager.max_subs else None
    minute = sim.match_minute()
    if new is not None:
        old = sim.players[weakest]
        sim.minutes_off[old.player_id] = minute
        sim.active[weakest] = False
        sim.events.append(sim.ev(0, weakest, E.SUBSTITUTION, "",
                                 52.5, 34.0, extra={"off": old.player_id, "on": new.player_id}))
        sim.put(gk_slot, new, minute)
        team.subs_used += 1
    else:
        pl = sim.players[weakest]
        sim.active[weakest] = False
        sim.players[gk_slot] = pl
        sim.base[gk_slot] = sim.base[weakest]
        sim.energy[gk_slot] = sim.energy[weakest]
        sim.active[gk_slot] = True
        sim.height[gk_slot] = sim.height[weakest]
    sim.refresh()
