"""目標點與無球意圖。

目標點：呼叫 v1 的陣型函式（縱向層次、橫向分布、盯人、逼搶位置），不使用 v1 的位置連續性與逼搶趕到距離上限——
位置由運動學決定（球員實際跑不跑得到），所以轉換期間的時間差是算出來的，不是被截斷的。
意圖：由階段、與目標點的距離、球的位置決定速度比例；DT 窗口內離球最近的人反搶。
"""
from __future__ import annotations

import numpy as np

from ..engine import shape as v1shape
from .phase import counter_press_slots
from .world import Intent, Mode, Phase, World


def update_targets(w: World) -> None:
    sim = w.sim
    T = w.poss
    owner = w.owner if (w.mode == Mode.CARRIED and w.owner is not None and w.team_of[w.owner] == T) else None
    b = w.tf(T, w.ball[:2])
    sim.gpos = None                       # 關掉 v1 的位置連續性：移動由 physics.integrate 負責
    sim.poss, sim.carrier, sim.ball = T, owner, (float(w.ball[0]), float(w.ball[1]))
    sim.elapsed = w.clock
    sim._hold = w.sp.shape.press_hold
    try:
        pos_T, off = v1shape.compute(sim, T, float(b[0]), float(b[1]), owner)
    finally:
        sim._hold = None
    tg = w.tf(T, pos_T.copy())
    act = sim.active
    # v1 的陣型每次重算都會重抽位置誤差與跑位；球員不會每秒重新抽籤，所以目標點做指數平滑。
    # 球權、階段改變與重啟時立即跳到新的目標（轉換要快）。
    a = 1.0 if (w.fast_shape or w.tgt_s is None) else w.sp.shape.smooth
    if w.tgt_s is None:
        w.tgt_s = tg.copy()
    w.tgt_s[act] += a * (tg[act] - w.tgt_s[act])
    w.fast_shape = False
    w.tgt[act] = w.tgt_s[act]
    w.off_line = float(off)
    sim.gpos = None


def assign_intents(w: World) -> None:
    sim = w.sim
    intent = np.full(22, int(Intent.HOLD))
    d_tgt = np.hypot(w.tgt[:, 0] - w.pos[:, 0], w.tgt[:, 1] - w.pos[:, 1])
    d_ball = np.hypot(w.pos[:, 0] - w.ball[0], w.pos[:, 1] - w.ball[1])
    for T in (0, 1):
        idx = np.array(w.active(T), int)
        if len(idx) == 0:
            continue
        ph = w.phase[T]
        far = d_tgt[idx] > 8.0
        if ph in (Phase.DT, Phase.OOP):
            intent[idx[far]] = Intent.RECOVER
            if ph == Phase.OOP:
                near = np.hypot(w.tgt[idx, 0] - w.ball[0], w.tgt[idx, 1] - w.ball[1]) < 5.0   # 陣型要他貼近持球者
                intent[idx[near]] = Intent.PRESS
        else:
            ahead = (w.tf(T, w.tgt[idx])[:, 0] - w.tf(T, w.pos[idx])[:, 0]) > 8.0
            intent[idx[ahead]] = Intent.RUN
        if ph == Phase.DT and w.mode != Mode.DEAD:
            for s in counter_press_slots(w, T):
                intent[s] = Intent.PRESS
                w.tgt[s] = w.ball[:2]
    if w.mode == Mode.FREE:
        for T in (0, 1):
            act = np.array(w.active(T), int)
            gk = sim.gk_slot(T)
            act = act[act != gk] if gk is not None else act
            for s in act[np.argsort(d_ball[act])[:2]]:
                intent[s] = Intent.CHASE
                w.tgt[s] = w.ball[:2] + w.bvel[:2] * 0.4
    w.intent = intent
    for s, (it, tgt) in list(w.override.items()):       # 接球、持球推進等優先
        w.intent[s] = int(it)
        w.tgt[s] = tgt
