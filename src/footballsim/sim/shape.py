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
    ball = w.ball[:2]
    f = w.flight
    if w.mode == Mode.FLIGHT and f is not None and f.kind in ("pass", "through", "cross") and f.receiver is not None:
        # 球在飛行中：防守者預判接球者與落點，提前往那裡逼近（陣型以「接球者在落點持球」計算）
        owner, ball, T = int(f.receiver), f.end, f.team
    b = w.tf(T, ball)
    sim.gpos = None                       # 關掉 v1 的位置連續性：移動由 physics.integrate 負責
    sim.poss, sim.carrier, sim.ball = T, owner, (float(ball[0]), float(ball[1]))
    sim.elapsed = w.clock
    sim._hold = w.sp.shape.press_hold
    # v1 的陣型每次呼叫都重抽位置誤差與跑位候選；真實球員的位置誤差是慢慢變化的。同一個時間窗內用同一組亂數，
    # 誤差就在窗內保持不變（窗與窗之間才換），目標點不會每秒抖動。
    sim.rng_shape = np.random.default_rng([w.seed, int(w.clock // w.sp.shape.noise_hold)])
    try:
        pos_T, off = v1shape.compute(sim, T, float(b[0]), float(b[1]), owner)
    finally:
        sim._hold = None
    tg = w.tf(T, pos_T.copy())
    act = sim.active
    # v1 的陣型每次重算都會重抽位置誤差與跑位；球員不會每秒重新抽籤，所以目標點做指數平滑。
    # 球權、階段改變與重啟時立即跳到新的目標（轉換要快）。
    # 球附近的人（逼搶、貼身盯人、接應）的目標點跟著球走，不平滑；離球遠的人才平滑
    if w.fast_shape or w.tgt_s is None:
        a = np.ones(22)
    else:
        near = np.hypot(tg[:, 0] - w.ball[0], tg[:, 1] - w.ball[1]) < w.sp.shape.near_ball
        a = np.where(near, 1.0, w.sp.shape.smooth)
    if w.tgt_s is None:
        w.tgt_s = tg.copy()
    w.tgt_s[act] += a[act, None] * (tg[act] - w.tgt_s[act])
    w.fast_shape = False
    w.tgt[act] = w.tgt_s[act]
    if w.mode == Mode.DEAD and w.restart is not None and w.restart["kind"] == "kickoff":
        for t in (0, 1):                                    # 開球前雙方都待在本方半場
            idx = np.array([s for s in w.active(t)], int)
            p = w.tf(t, w.tgt[idx])
            p[:, 0] = np.minimum(p[:, 0], 51.5)
            w.tgt[idx] = w.tf(t, p)
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
                dball = np.hypot(w.tgt[idx, 0] - w.ball[0], w.tgt[idx, 1] - w.ball[1])
                near = dball < 5.0                                   # 陣型要他貼近持球者
                intent[idx[near]] = Intent.PRESS
                # 持球者附近的盯人與補位要跟得上進攻者的跑動，不能只用保持陣型的慢速
                zone = (dball < w.sp.shape.defend_zone) & (d_tgt[idx] > 1.0) & ~far
                intent[idx[zone]] = Intent.PRESS

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
