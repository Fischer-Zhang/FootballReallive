"""裁判：犯規、出牌、紀律（§4.16）"""
from __future__ import annotations

import numpy as np

from ..model.attributes import ATTR
from . import events as E
from . import manager
from .core_math import sigmoid
from .pitch import CY, PEN_SPOT, in_box
from .state import Outcome, Restart


def _foul_logit(sim, k: int, x: float, y: float) -> float:
    """犯規者 k 的共同項：侵略性、骯髒度、疲勞、戰術、裁判、主場哨、已吃牌、禁區內收斂"""
    fp = sim.p.foul
    e = sim.eff
    D = k // 11
    return (fp.k_aggr * (e[k, ATTR.AGGRESSION] - 50) / 10
            + fp.k_dirty * (e[k, ATTR.DIRTINESS] - 50) / 10
            + fp.k_fatigue * (1.0 - sim.energy[k])
            + getattr(fp.style, sim.teams[D].tactics.tackling)
            + sim.ref_strict + sim.ref_home_bias[D]
            + (fp.booked if sim.yellow[k] else 0.0)
            + fp.dives_trait * sim.traits[k].get("dives_into_tackles", 0.0)
            + (fp.box if in_box(x, y) else 0.0))


def foul_check(sim, k: int, c: int, att_comp: float, def_comp: float, beaten: bool,
               x: float, y: float) -> bool:
    """防守者 k 在搶斷／過人對抗中是否犯規；(x, y) 為進攻方座標"""
    fp = sim.p.foul
    z = (fp.b0 + fp.k_skill * (att_comp - def_comp) / 10 + _foul_logit(sim, k, x, y)
         + (fp.beaten if beaten else 0.0))
    return sim.rng_ref.random() < sigmoid(z)


def receive_foul(sim, k: int, rp: float, lofted: bool, x: float, y: float) -> bool:
    """接球受壓時的身體對抗犯規（含高球爭頂）"""
    fp = sim.p.foul
    z = fp.recv_b0 + fp.recv_k * rp + (fp.aerial if lofted else 0.0) + _foul_logit(sim, k, x, y)
    return sim.rng_ref.random() < sigmoid(z)


def attacker_foul(sim, T: int, j: int, k: int, x: float, y: float, dur: float, evs: list) -> Outcome:
    """進攻方犯規（爭頂推人、背身撞人）→ 防守方自由球"""
    evs.append(sim.ev(T, j, E.FOUL, E.SUCCESS, x, y, opponent=sim.pid(k)))
    e = sim.eff
    cp = sim.p.card
    z = (cp.yellow_b0 - 1.0 + cp.k_aggr * (e[j, ATTR.AGGRESSION] - 50) / 10
         + sim.ref_strict + sim.ref_home_bias[T])
    if sim.rng_ref.random() < sigmoid(z):
        evs.extend(give_card(sim, j, "yellow", T, x, y))
    return Outcome(duration=dur, events=evs,
                   restart=Restart("free_kick", 1 - T, *sim.to_global(T, x, y)))


def award_foul(sim, T: int, c: int, k: int, x: float, y: float, pos, dur: float, evs: list,
               beaten: bool) -> Outcome:
    evs.append(sim.ev(T, k, E.FOUL, E.SUCCESS, x, y, opponent=sim.pid(c)))
    box = in_box(x, y)
    card = decide_card(sim, T, k, x, y, pos, beaten, box)
    if card:
        evs.extend(give_card(sim, k, card, T, x, y))
    manager.contact_injury(sim, c, evs, T, x, y)
    if box:
        restart = Restart("penalty", T, *sim.to_global(T, *PEN_SPOT))
    else:
        restart = Restart("free_kick", T, *sim.to_global(T, x, y))
    return Outcome(duration=dur, events=evs, restart=restart)


def decide_card(sim, T: int, k: int, x: float, y: float, pos, beaten: bool, box: bool):
    cp, rng, e = sim.p.card, sim.rng_ref, sim.eff
    D = 1 - T
    covering = sum(1 for s in sim.active_slots(D)
                   if s != k and not sim.is_gk(s) and pos[s, 0] > x)
    # DOGSO：破壞明顯進球機會；禁區內且有爭球意圖時多半降為黃牌（IFAB）
    if beaten and x >= 80 and abs(y - CY) <= 18 and covering == 0:
        p_red = cp.dogso_box_red if box else cp.dogso_red
        return "red" if rng.random() < p_red else "yellow"
    spa = beaten and x >= 50 and covering <= 3        # 阻止有希望的進攻
    if rng.random() < sigmoid(cp.red_b0 + 0.2 * (e[k, ATTR.AGGRESSION] - 50) / 10
                              + 0.2 * (50 - e[k, ATTR.TEMPERAMENT]) / 10):
        return "red"
    z = (cp.yellow_b0 + cp.k_aggr * (e[k, ATTR.AGGRESSION] - 50) / 10
         + cp.k_dirty * (e[k, ATTR.DIRTINESS] - 50) / 10
         + cp.k_tackle * (65 - e[k, ATTR.TACKLING]) / 10
         + (cp.spa if spa else 0.0) + sim.ref_strict
         + sim.ref_home_bias[D] * sim.p.referee.card_bias_mult)
    p_y = sigmoid(z)
    if sim.yellow[k] >= 1:
        # 已吃黃牌：球員收斂動作、裁判也較謹慎（以真實「兩黃變紅 ÷ 黃牌」的比例校準）
        p_y *= cp.booked_factor
    return "yellow" if rng.random() < p_y else None


def give_card(sim, s: int, card: str, frame: int, x: float, y: float) -> list:
    evs = []
    if card == "yellow" and sim.yellow[s] >= 1:
        evs.append(sim.ev(frame, s, E.CARD, E.SECOND_YELLOW, x, y))
        send_off(sim, s)
    elif card == "yellow":
        sim.yellow[s] += 1
        evs.append(sim.ev(frame, s, E.CARD, E.YELLOW, x, y))
    else:
        evs.append(sim.ev(frame, s, E.CARD, E.RED, x, y))
        send_off(sim, s)
    sim.extra_dead[sim.period - 1] += sim.p.stoppage.card
    return evs


def send_off(sim, s: int) -> None:
    sim.active[s] = False
    pl = sim.players[s]
    if pl is not None:
        sim.minutes_off[pl.player_id] = sim.match_minute()
    t = s // 11
    sim.teams[t].red_cards += 1
    if sim.is_gk(s):
        manager.replace_goalkeeper(sim, t)


def minute_hazards(sim) -> None:
    """非犯規類黃牌：抗議（脾氣）、領先時拖延（運動精神）"""
    cp, rng = sim.p.card, sim.rng_ref
    minute = sim.match_minute()
    for t in (0, 1):
        slots = sim.active_slots(t)
        if not slots:
            continue
        temper = sim.eff[slots, ATTR.TEMPERAMENT]
        p = cp.dissent_per_min * float(np.mean(1.0 + (50 - temper) / 50))
        if minute >= 75 and sim.teams[t].score > sim.teams[1 - t].score:
            sport = sim.eff[slots, ATTR.SPORTSMANSHIP]
            p += cp.time_wasting_per_min * float(np.mean(1.0 + (50 - sport) / 50))
        if rng.random() < p:
            w = np.maximum(1.0, 110.0 - temper)
            s = slots[int(rng.choice(len(slots), p=w / w.sum()))]
            if s == sim.carrier and sim.yellow[s] >= 1:
                continue      # 不在持球中途罰下持球者
            sim.events.extend(give_card(sim, s, "yellow", 0, *sim.to_frame(0, *sim.ball)))
