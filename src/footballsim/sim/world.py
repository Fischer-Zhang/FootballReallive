"""世界狀態：22 名球員、球、兩隊的階段、時鐘與事件。

座標一律是全域座標（主隊 = team 0 往 +x 進攻）；呼叫 v1 的決策與陣型函式時，用 tf() 轉成持球隊的進攻座標系。
球員 slot 0–10 為主隊、11–21 為客隊（與 v1 相同）。屬性、參數、亂數流與當日狀態由內含的 v1 MatchSim 提供。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from ..engine.events import Event
from ..engine.match import MatchSim
from ..engine.pitch import LENGTH, WIDTH
from .params import load_sim_params


class Mode(IntEnum):
    CARRIED = 0       # 球在持球者腳下
    FLIGHT = 1        # 傳球、射門等飛行中
    FREE = 2          # 無主球（滾動）
    DEAD = 3          # 死球（出界、進球後、等待重啟）


class Phase(IntEnum):
    IP = 0            # 持球進攻
    DT = 1            # 防守轉換（剛失球）
    OOP = 2           # 無球防守
    AT = 3            # 進攻轉換（剛得球）


class Intent(IntEnum):
    HOLD = 0          # 保持陣型位置
    RUN = 1           # 前插／跑位
    PRESS = 2         # 逼搶
    RECOVER = 3       # 回防
    RECEIVE = 4       # 去接球
    CHASE = 5         # 搶無主球
    CARRY = 6         # 持球推進
    WAIT = 7          # 持球等待出手（追蹤資料：持球者接球後不會站著，速度中位數約 2–3 m/s）
    ENGAGE = 8        # 離球最近的防守者貼近持球者（追蹤資料：最近的對手速度中位數約 4 m/s，不論距離）


@dataclass
class Flight:
    kind: str                     # pass / cross / through / clearance / shot
    kicker: int
    team: int
    start: np.ndarray             # (2,) 全域
    end: np.ndarray               # (2,) 實際落點（含執行誤差）
    t0: float
    dur: float
    v0: float
    lofted: bool = False
    hmax: float = 0.0
    receiver: Optional[int] = None
    aim: Optional[np.ndarray] = None   # 原本瞄準的點（事件紀錄用）
    tried: set = field(default_factory=set)   # 已嘗試攔截的球員
    shot: Optional[dict] = None   # 射門的預定結果：{"result": goal/saved/blocked/off, "xg": ...}
    decel: float = 0.0
    event: Optional[dict] = None  # 待補結果的事件資訊


class World:
    def __init__(self, home, away, params=None, sim_params=None, seed: int = 0):
        from ..config import load_params
        params = params or load_params()                      # 每個 v2 世界用自己的參數物件，改它不會影響 v1 的全域預設
        self.sim = MatchSim(home, away, params, seed=seed)     # v1：屬性、參數、亂數流、威脅網格
        self.sp = sim_params or load_sim_params()
        # v1 的橫向分位數抽籤每次決策漂移 0.25（約 3 m）；v2 每秒呼叫一次且球員實際要跑過去，寬位球員跟不上這種漂移，
        # 結果都縮到中路。位置的橫向漂移改用較慢的速度。
        params.shape.att_lateral_drift = self.sp.shape.lateral_drift
        self.seed = seed
        n = 22
        self.team_of = np.repeat([0, 1], 11)
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.tgt = np.zeros((n, 2))
        self.intent = np.full(n, int(Intent.HOLD))
        self.stun = np.zeros(n)               # 被過掉後暫時不能搶斷的剩餘秒數
        self.dist_run = np.zeros(n)           # 跑動距離（m）
        self.ball = np.array([LENGTH / 2, WIDTH / 2, 0.0])
        self.bvel = np.zeros(3)
        self.mode = Mode.DEAD
        self.owner: Optional[int] = None
        self.flight: Optional[Flight] = None
        self.last_touch: Optional[int] = None
        self.poss = 0                         # 控球隊（球在飛行或無主時為最後觸球隊）
        self.phase = [Phase.IP, Phase.OOP]
        self.phase_t = [0.0, 0.0]
        self.counter_from: list = [None, None]      # 進攻轉換開始時的球位置（偵測反擊）
        self.t = 0.0                          # 半場內秒數
        self.clock = 0.0                      # 全場累計秒數
        self.period = 1
        self.tick_n = 0
        self.carry: Optional[dict] = None     # 持球者目前的動作狀態
        self.restart: Optional[dict] = None
        self.score = [0, 0]
        self.ht_score = (0, 0)
        self.events: list = []
        self.bip = 0.0
        self.bip_half = [0.0, 0.0]
        self.poss_time = [0.0, 0.0]
        self.next_shape = 0.0
        self.next_intent = 0.0
        self.next_refresh = self.sp.motion.refresh_every
        self.off_line = 52.5                  # 控球隊座標系下的越位線（shape 更新時一併更新）
        self.stoppage = [0, 0]
        self.tgt_s = None                     # 平滑後的目標點
        self.fast_shape = False               # 下一次目標點更新不平滑（球權改變、重啟）
        self.override: dict = {}              # slot → (意圖, 目標點)：接球、搶球、持球推進等優先於陣型

    # ------------------------------------------------------------------ 座標
    @staticmethod
    def tf(T: int, a: np.ndarray) -> np.ndarray:
        """全域 ↔ 隊 T 的進攻座標系（互為反函數）"""
        if T == 0:
            return np.array(a, float)
        out = np.array(a, float)
        out[..., 0] = LENGTH - out[..., 0]
        out[..., 1] = WIDTH - out[..., 1]
        return out

    # ------------------------------------------------------------------ 事件
    def pid(self, s: Optional[int]) -> Optional[str]:
        return self.sim.pid(s) if s is not None else None

    def add_event(self, slot: Optional[int], etype: str, result: str, p0, p1=None, team: Optional[int] = None,
                  **kw) -> Event:
        p1 = p0 if p1 is None else p1
        tm = team if team is not None else (int(self.team_of[slot]) if slot is not None else self.poss)
        ev = Event(self.period, self.t, tm, self.pid(slot), etype, result,
                   float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), **kw)
        self.events.append(ev)
        return ev

    def active(self, T: int) -> list:
        return self.sim.active_slots(T)
