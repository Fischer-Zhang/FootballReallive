"""球場幾何與價值網格（§4.4）。

座標：105 × 68 m。引擎內部一律使用「持球隊進攻方向 = +x」的座標系；
全域座標（事件紀錄）以主隊（team 0）進攻方向為 +x。flip() 在兩者間轉換。
"""
from __future__ import annotations

import math

import numpy as np

from .core_math import base_xg

LENGTH = 105.0
WIDTH = 68.0
CY = 34.0
GOAL_HALF = 3.66
BOX_X = 88.5
BOX_HALF = 20.16
SIX_X = 99.5
SIX_HALF = 9.16
PEN_SPOT = (94.0, 34.0)


def flip(x: float, y: float) -> tuple[float, float]:
    return LENGTH - x, WIDTH - y


def in_box(x: float, y: float) -> bool:
    return x >= BOX_X and abs(y - CY) <= BOX_HALF


def dist_to_goal(x: float, y: float) -> float:
    return math.hypot(LENGTH - x, y - CY)


CONE_MARGIN = 0.08   # rad：射門角度兩側的寬容（約一個身體寬）
NEAR_CAP = 8.0       # 最近防守者距離的上限（m）


def shot_context(bx: float, by: float, defenders: np.ndarray):
    """射門情境特徵（引擎與 StatsBomb 擬合共用同一個定義）。

    defenders：對方外野球員座標 (n, 2)，不含門將；座標為射門方的進攻座標系。
    回傳 (射門角度內人數, 最近防守者距離[上限 8 m], 角度內的布林遮罩)。
    """
    if len(defenders) == 0:
        return 0, NEAR_CAP, np.zeros(0, dtype=bool)
    dist_goal = math.hypot(LENGTH - bx, by - CY)
    a1 = math.atan2(CY - GOAL_HALF - by, LENGTH - bx)
    a2 = math.atan2(CY + GOAL_HALF - by, LENGTH - bx)
    lo, hi = min(a1, a2) - CONE_MARGIN, max(a1, a2) + CONE_MARGIN
    rel = np.asarray(defenders, float) - (bx, by)
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    dk = np.hypot(rel[:, 0], rel[:, 1])
    inc = (rel[:, 0] > 0) & (ang >= lo) & (ang <= hi) & (dk < dist_goal)
    return int(inc.sum()), float(min(dk.min(), NEAR_CAP)), inc


class Threat:
    """xT（解析式起始版）與 xG 查表，1 m 解析度。

    xT(x, y) = v0·exp(a·x/105)·(1 − lateral·((y−34)/34)²) + w_xg·xG_foot(x, y)
    數值量級參照公開 xT 熱圖：本方半場約 0.005–0.02、禁區前沿約 0.1、十二碼點約 0.17。
    """

    def __init__(self, threat: dict, xg: dict):
        nx, ny = int(LENGTH) + 1, int(WIDTH) + 1
        self.xg_foot = np.zeros((nx, ny))
        self.xg_head = np.zeros((nx, ny))
        for i in range(nx):
            gx = max(LENGTH - i, 0.5)
            for j in range(ny):
                gy = abs(j - CY)
                self.xg_foot[i, j] = base_xg(gx, gy, False, xg)
                self.xg_head[i, j] = base_xg(gx, gy, True, xg)
        if threat.get("source") == "grid":
            self.xt = _grid_to_meters(_load_grid(threat.get("grid_file", "xt_grid.yaml")), nx, ny)
        else:
            X = np.arange(nx)[:, None]
            Y = np.arange(ny)[None, :]
            prog = threat["v0"] * np.exp(threat["a"] * X / LENGTH) \
                * (1 - threat["lateral"] * ((Y - CY) / CY) ** 2)
            self.xt = prog + threat["w_xg"] * self.xg_foot

    @staticmethod
    def _ij(x: float, y: float) -> tuple[int, int]:
        i = int(min(max(x, 0.0), LENGTH) + 0.5)
        j = int(min(max(y, 0.0), WIDTH) + 0.5)
        return i, j

    def xt_at(self, x: float, y: float) -> float:
        i, j = self._ij(x, y)
        return float(self.xt[i, j])

    def xt_opp(self, x: float, y: float) -> float:
        """對手在 (x, y)（本隊座標）拿到球時的威脅"""
        return self.xt_at(LENGTH - x, WIDTH - y)

    def xt_many(self, pts: np.ndarray) -> np.ndarray:
        i = np.clip(np.rint(pts[..., 0]), 0, LENGTH).astype(np.intp)
        j = np.clip(np.rint(pts[..., 1]), 0, WIDTH).astype(np.intp)
        return self.xt[i, j]

    def xt_opp_many(self, pts: np.ndarray) -> np.ndarray:
        i = np.clip(np.rint(LENGTH - pts[..., 0]), 0, LENGTH).astype(np.intp)
        j = np.clip(np.rint(WIDTH - pts[..., 1]), 0, WIDTH).astype(np.intp)
        return self.xt[i, j]

    def xg_at(self, x: float, y: float, header: bool = False) -> float:
        i, j = self._ij(x, y)
        return float((self.xg_head if header else self.xg_foot)[i, j])


def _load_grid(name: str) -> np.ndarray:
    import yaml

    from ..config import CONFIG_DIR

    doc = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    return np.array(doc["values"], dtype=float)


def _grid_to_meters(grid: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """以格子中心做雙線性內插，得到每公尺一格的 xT"""
    gx, gy = grid.shape
    fx = np.clip(np.arange(nx) / LENGTH * gx - 0.5, 0, gx - 1)
    fy = np.clip(np.arange(ny) / WIDTH * gy - 0.5, 0, gy - 1)
    i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    i1, j1 = np.minimum(i0 + 1, gx - 1), np.minimum(j0 + 1, gy - 1)
    wx, wy = (fx - i0)[:, None], (fy - j0)[None, :]
    return ((1 - wx) * (1 - wy) * grid[np.ix_(i0, j0)] + wx * (1 - wy) * grid[np.ix_(i1, j0)]
            + (1 - wx) * wy * grid[np.ix_(i0, j1)] + wx * wy * grid[np.ix_(i1, j1)])


_CACHE: dict = {}


def threat_for(params) -> Threat:
    key = (tuple(sorted(params._raw["threat"].items())), tuple(sorted(params._raw["xg"].items())))
    if params._raw["threat"].get("source") == "grid":
        from ..config import CONFIG_DIR

        key += ((CONFIG_DIR / params._raw["threat"].get("grid_file", "xt_grid.yaml")).stat().st_mtime,)
    if key not in _CACHE:
        _CACHE[key] = Threat(params._raw["threat"], params._raw["xg"])
    return _CACHE[key]
