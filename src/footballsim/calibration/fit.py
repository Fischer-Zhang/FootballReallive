"""以 StatsBomb 事件資料擬合引擎基準（M4，設計文件 §6.2、§6.4 L1）

產出：
- config/xt_grid.yaml               16×12 Expected Threat 網格（Karun Singh 迭代法）
- config/targets/event_level.yaml   事件層校準目標（取代原本的概略範圍）
- xG 模型係數（距離、ln 夾角、頭槌）與傳球成功率曲線（報告用，係數再寫入 engine_params）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..config import CONFIG_DIR
from ..engine.core_math import shot_geometry

NX, NY = 16, 12
LENGTH, WIDTH = 105.0, 68.0
SET_PIECES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
SOURCE = "StatsBomb Open Data（2015/16 五大聯賽完整賽季，隨機抽樣）"


def _cell(x, y) -> np.ndarray:
    i = np.clip((np.asarray(x, float) / LENGTH * NX).astype(int), 0, NX - 1)
    j = np.clip((np.asarray(y, float) / WIDTH * NY).astype(int), 0, NY - 1)
    return i * NY + j


# ------------------------------------------------------------------ xT
def expected_threat(df, n_iter: int = 60) -> tuple[np.ndarray, dict]:
    """xT(z) = s(z)·g(z) + m(z)·Σ T(z→z')·xT(z')。運動戰的傳球與帶球為移動，運動戰射門為射門；
    g(z) 用該格射門的平均 StatsBomb xG（比進球數穩定）。"""
    moves = df[df.type.isin(["pass", "carry"]) & ~df["sub"].isin(SET_PIECES)]
    shots = df[(df.type == "shot") & (df["sub"] == "Open Play")]
    n = NX * NY
    z_m = _cell(moves.x, moves.y)
    z_s = _cell(shots.x, shots.y)
    n_move = np.bincount(z_m, minlength=n).astype(float)
    n_shot = np.bincount(z_s, minlength=n).astype(float)
    xg_sum = np.bincount(z_s, weights=shots.xg.astype(float), minlength=n)
    g = np.divide(xg_sum, n_shot, out=np.zeros(n), where=n_shot > 0)
    tot = n_move + n_shot
    s = np.divide(n_shot, tot, out=np.zeros(n), where=tot > 0)
    m = np.divide(n_move, tot, out=np.zeros(n), where=tot > 0)
    ok = moves[(moves.ok == 1) & moves.ex.notna()]
    T = np.zeros((n, n))
    np.add.at(T, (_cell(ok.x, ok.y), _cell(ok.ex, ok.ey)), 1.0)
    T = np.divide(T, n_move[:, None], out=np.zeros_like(T), where=n_move[:, None] > 0)
    xt = np.zeros(n)
    for _ in range(n_iter):
        xt = s * g + m * (T @ xt)
    info = {"moves": int(len(moves)), "shots": int(len(shots)), "empty_cells": int((tot == 0).sum())}
    return xt.reshape(NX, NY), info


def save_xt(grid: np.ndarray, info: dict, path: Path = CONFIG_DIR / "xt_grid.yaml") -> Path:
    doc = {"source": SOURCE, "nx": NX, "ny": NY,
           "note": "列 = x 方向 16 格（本方球門 → 對方球門），欄 = y 方向 12 格（右路 → 左路）",
           "info": info, "values": [[round(float(v), 5) for v in row] for row in grid]}
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 邏輯斯迴歸（IRLS，無需 scipy）
def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-4, n_iter: int = 100,
                 sw: np.ndarray | None = None) -> np.ndarray:
    """牛頓法的邏輯迴歸；sw 為各樣本的權重（例如校正樣本選擇偏誤的反向抽樣權重）"""
    w = np.zeros(X.shape[1])
    sw = np.ones(len(y)) if sw is None else np.asarray(sw, float)
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-(X @ w)))
        W = p * (1 - p) * sw
        H = X.T @ (X * W[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, X.T @ (sw * (y - p)) - l2 * w)
        w += step
        if np.abs(step).max() < 1e-9:
            break
    return w


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_xg(df, current: dict | None = None) -> dict:
    """運動戰射門（不含十二碼、直接自由球）：logit = b0 + b_dist·距離 + b_logang·ln(夾角) + b_head·頭槌"""
    s = df[(df.type == "shot") & (df["sub"] == "Open Play")].copy()
    geo = [shot_geometry(max(105 - x, 0.3), abs(y - 34)) for x, y in zip(s.x, s.y)]
    dist = np.array([g[0] for g in geo])
    lang = np.log(np.maximum([g[1] for g in geo], 1e-4))
    head = (s.body == "Head").to_numpy(float)
    X = np.column_stack([np.ones(len(s)), dist, lang, head])
    y = s.ok.to_numpy(float)
    w = logistic_fit(X, y)
    out = {"n_shots": int(len(s)), "goals": int(y.sum()),
           "coef": {"b0": round(float(w[0]), 4), "b_dist": round(float(w[1]), 4),
                    "b_logang": round(float(w[2]), 4), "b_head": round(float(w[3]), 4)},
           "log_loss_fit": log_loss(1 / (1 + np.exp(-(X @ w))), y),
           "log_loss_statsbomb_xg": log_loss(s.xg.to_numpy(float), y),
           "log_loss_constant": log_loss(np.full(len(y), y.mean()), y)}
    if current:
        c = np.array([current["b0"], current["b_dist"], current["b_logang"], current["b_head"]])
        out["log_loss_current"] = log_loss(1 / (1 + np.exp(-(X @ c))), y)
    return out


# ------------------------------------------------------------------ 傳球成功率
def pass_curves(df) -> dict:
    p = df[(df.type == "pass") & ~df["sub"].isin(SET_PIECES) & df.ex.notna()].copy()
    p["dist"] = np.hypot(p.ex - p.x, p.ey - p.y)
    p["lofted"] = p.height.isin(["Low Pass", "High Pass"])
    bins = [0, 10, 20, 30, 40, 60, 120]
    out = {"overall%": round(100 * p.ok.mean(), 1),
           "under_pressure%": round(100 * p[p.pressure == 1].ok.mean(), 1),
           "no_pressure%": round(100 * p[p.pressure == 0].ok.mean(), 1), "by_distance": {}}
    for kind, sub in (("ground", p[~p.lofted]), ("lofted", p[p.lofted])):
        idx = np.digitize(sub.dist, bins) - 1
        out["by_distance"][kind] = {
            f"{bins[i]}-{bins[i + 1]}m": [round(100 * float(sub.ok[idx == i].mean()), 1), int((idx == i).sum())]
            for i in range(len(bins) - 1) if (idx == i).sum() >= 30}
        X = np.column_stack([np.ones(len(sub)), sub.dist, sub.pressure])
        w = logistic_fit(X, sub.ok.to_numpy(float))
        out[f"{kind}_logit"] = {"b0": round(float(w[0]), 3), "b_dist": round(float(w[1]), 4),
                                "b_pressure": round(float(w[2]), 3)}
    return out


# ------------------------------------------------------------------ 事件層目標
def _team_match(df, mask, weights=None):
    """每隊每場的平均與標準誤（以 match×team 為單位）"""
    g = df[mask].groupby(["match_id", "team"]).size() if weights is None else \
        df[mask].groupby(["match_id", "team"])[weights].sum()
    teams = df.groupby(["match_id", "team"]).size().index
    v = g.reindex(teams, fill_value=0).to_numpy(float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def event_targets(df) -> dict:
    n_matches = df.match_id.nunique()
    t = {}
    passes = (df.type == "pass") & (df["sub"] != "Throw-in")
    t["passes_pt"] = _team_match(df, passes)
    pa = df[passes]
    t["pass_pct"] = (100 * pa.ok.mean(), 100 * np.sqrt(pa.ok.mean() * (1 - pa.ok.mean()) / len(pa)))
    open_pass = (df.type == "pass") & ~df["sub"].isin(SET_PIECES)
    cr = df[open_pass & (df.cross == 1)]
    t["crosses_pt"] = _team_match(df, open_pass & (df.cross == 1))
    t["cross_pct"] = (100 * cr.ok.mean(), 100 * np.sqrt(cr.ok.mean() * (1 - cr.ok.mean()) / len(cr)))
    to = df[df.type == "take_on"]
    t["take_ons_pt"] = _team_match(df, df.type == "take_on")
    t["take_on_pct"] = (100 * to.ok.mean(), 100 * np.sqrt(to.ok.mean() * (1 - to.ok.mean()) / len(to)))
    sh = df[df.type == "shot"]
    t["blocked_pct"] = (100 * (sh.shot_out == "Blocked").mean(),
                        100 * np.sqrt((sh.shot_out == "Blocked").mean() * (1 - (sh.shot_out == "Blocked").mean()) / len(sh)))
    t["xg_per_shot"] = (float(sh.xg.mean()), float(sh.xg.std() / np.sqrt(len(sh))))
    goals = sh[sh.ok == 1]
    hg = (goals.body == "Head").mean()
    t["header_goal_share%"] = (100 * hg, 100 * np.sqrt(hg * (1 - hg) / len(goals)))
    pens = (sh["sub"] == "Penalty").sum() / n_matches
    t["penalties_pm"] = (float(pens), float(np.sqrt(pens / n_matches)))
    # 越位：傳球結果為 Pass Offside（parse_events 把傳球結果存在 shot_out 欄）
    t["offsides_pt"] = _team_match(df, (df.type == "pass") & (df["shot_out"].astype(str) == "Pass Offside"))
    np_shots = sh[sh["sub"] != "Penalty"]
    d = np.hypot(105 - np_shots.x, np_shots.y - 34)
    hist = np.histogram(d, [0, 6, 11, 16.5, 25, 200])[0] / len(d) * 100
    t["shot_dist_%"] = {k: round(float(v), 1) for k, v in
                        zip(["0-6", "6-11", "11-16.5", "16.5-25", "25+"], hist)}
    t["_matches"] = int(n_matches)
    t["_goals_pm"] = float((sh.ok == 1).sum() / n_matches)
    return t


def save_event_targets(t: dict, path: Path = CONFIG_DIR / "targets" / "event_level.yaml") -> Path:
    """範圍 = 擬合值 ± max(2 個標準誤, 相對容忍度)，讓範圍中點 = 實測值"""
    rel = {"passes_pt": 0.08, "pass_pct": 0.025, "crosses_pt": 0.12, "cross_pct": 0.10,
           "take_ons_pt": 0.15, "take_on_pct": 0.08, "blocked_pct": 0.10, "xg_per_shot": 0.08,
           "header_goal_share%": 0.20, "penalties_pm": 0.25, "offsides_pt": 0.20}
    doc = {}
    for k, r in rel.items():
        v, se = float(t[k][0]), float(t[k][1])
        half = max(2 * se, abs(v) * r)
        doc[k] = [round(v - half, 3), round(v + half, 3)]
    for name, v in t["shot_dist_%"].items():   # 射門距離分布（不含十二碼）
        v = float(v)
        half = max(2.5, 0.15 * v)
        doc[f"shot_{name}%"] = [round(v - half, 1), round(v + half, 1)]
    keep = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    for k in ("bip_min", "subs_pt"):          # StatsBomb 無法計算者沿用原概略範圍
        if k in keep:
            doc[k] = keep[k]
    header = (f"# 事件層校準目標（設計文件 §6.2）\n# 來源：{SOURCE}，{t['_matches']} 場；"
              "範圍 = 實測值 ± max(2 個標準誤, 相對容忍度)。\n"
              "# 注意：2015/16 的傳球成功率比近年低約 3–4 個百分點；bip_min、subs_pt 仍為概略範圍。\n")
    path.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 射門情境模型（freeze frame）
def fit_shot_models(shots) -> dict:
    """運動戰射門的三個模型，特徵與引擎共用 pitch.shot_context 的定義：
    進球：logit = b0 + 距離 + ln(夾角) + 頭槌 + 角度內防守人數 + 最近防守者距離
    封堵：logit = c0 + 角度內人數 + 最近防守者距離 + 距離 + 頭槌
    射正（未被封堵時）：logit = e0 + 距離 + 頭槌 + 最近防守者距離"""
    s = shots[shots.type == "Open Play"].copy()
    geo = [shot_geometry(max(105 - x, 0.3), abs(y - 34)) for x, y in zip(s.x, s.y)]
    dist = np.array([g[0] for g in geo])
    lang = np.log(np.maximum([g[1] for g in geo], 1e-4))
    head, cone, near = s["head"].to_numpy(float), s.n_cone.to_numpy(float), s.d_near.to_numpy(float)
    one = np.ones(len(s))
    Xg = np.column_stack([one, dist, lang, head, cone, near])
    yg = s.goal.to_numpy(float)
    wg = logistic_fit(Xg, yg)
    Xb = np.column_stack([one, cone, near, dist, head])
    wb = logistic_fit(Xb, s.blocked.to_numpy(float))
    ub = s.blocked.to_numpy() == 0
    Xo = np.column_stack([one, dist, head, near])[ub]
    wo = logistic_fit(Xo, s.on_target.to_numpy(float)[ub])
    pg = 1 / (1 + np.exp(-(Xg @ wg)))

    def r(w, names):
        return {n: round(float(v), 4) for n, v in zip(names, w)}

    return {"n_shots": int(len(s)),
            "goal": r(wg, ["b0", "b_dist", "b_logang", "b_head", "b_cone", "b_near"]),
            "block": r(wb, ["c0", "c_cone", "c_near", "c_dist", "c_head"]),
            "on_target": r(wo, ["e0", "e_dist", "e_head", "e_near"]),
            "log_loss": {"model": log_loss(pg, yg), "statsbomb_xg": log_loss(s.xg.to_numpy(float), yg),
                         "constant": log_loss(np.full(len(yg), yg.mean()), yg)},
            "rates": {"blocked%": round(100 * float(s.blocked.mean()), 1),
                      "on_target%": round(100 * float(s.on_target.mean()), 1),
                      "goal%": round(100 * float(yg.mean()), 1)}}


# ------------------------------------------------------------------ 轉移價值（事件序列）
_CONTROLLED = {"pass", "carry", "take_on", "shot"}


def _xt_lookup(grid: np.ndarray, x, y) -> np.ndarray:
    i = np.clip((np.asarray(x, float) / LENGTH * NX).astype(int), 0, NX - 1)
    j = np.clip((np.asarray(y, float) / WIDTH * NY).astype(int), 0, NY - 1)
    return grid[i, j]


def transition_values(df, grid: np.ndarray) -> dict:
    """傳中失敗、射門未進、解圍之後「下一次有人控球」的結果機率與價值。
    事件座標以執行動作的隊伍為進攻方向，所以對手的 xT 可直接查表。"""
    out = {}
    df = df.sort_values(["match_id", "period", "idx"]).reset_index(drop=True)
    ctrl = df.type.isin(_CONTROLLED).to_numpy()
    team = df.team.to_numpy()
    mid = df.match_id.to_numpy()
    sub = df["sub"].astype(str).to_numpy()
    nxt = np.full(len(df), -1)
    last = -1
    for i in range(len(df) - 1, -1, -1):          # 每列之後第一個「有人控球」的事件
        nxt[i] = last if last >= 0 and mid[last] == mid[i] else -1
        if ctrl[i]:
            last = i

    def outcome(mask) -> dict:
        idx = np.flatnonzero(mask)
        j = nxt[idx]
        ok = j >= 0
        idx, j = idx[ok], j[ok]
        corner = (sub[j] == "Corner") & (team[j] == team[idx])
        retain = (team[j] == team[idx]) & ~corner
        lose = team[j] != team[idx]
        xj, yj = df.x.to_numpy()[j], df.y.to_numpy()[j]
        v = _xt_lookup(grid, xj, yj)
        return {"n": int(len(idx)), "p_corner": round(float(corner.mean()), 4),
                "p_retain": round(float(retain.mean()), 4), "p_lose": round(float(lose.mean()), 4),
                "v_retain": round(float(v[retain].mean()) if retain.any() else 0.0, 5),
                "v_lose": round(float(v[lose].mean()) if lose.any() else 0.0, 5)}

    open_cross = (df.type == "pass") & (df.cross == 1) & ~df["sub"].isin(SET_PIECES) & (df.ok == 0)
    out["cross_fail"] = outcome(open_cross.to_numpy())
    out["shot_miss"] = outcome(((df.type == "shot") & (df.ok == 0) & (df["sub"] == "Open Play")).to_numpy())
    out["clearance"] = outcome((df.type == "clearance").to_numpy())
    corners = int(((df.type == "pass") & (df["sub"] == "Corner")).sum())
    fc = df[(df.type == "shot") & (df.pattern == "From Corner")]
    out["corner_value"] = round(float(fc.xg.sum() / max(corners, 1)), 5)
    out["corner_goal_rate"] = round(float(fc.ok.sum() / max(corners, 1)), 5)
    return out


def save_fitted(shot_models: dict, trans: dict, typical: dict | None = None,
                path: Path = CONFIG_DIR / "fitted.yaml") -> Path:
    """2015/16 事件資料擬合的參數，載入時覆蓋 engine_params.yaml 的同名項目。
    行為模型（射門／傳中／解圍傾向）改用 360 位置特徵擬合，見 fit360.py。"""
    doc = {"_source": SOURCE,
           "shot_model": {"goal": shot_models["goal"], "block": shot_models["block"],
                          "on_target": shot_models["on_target"]},
           "transitions": trans}
    if typical:
        doc["shot_model"]["typical"] = typical
    path.write_text("# 由 `python -m footballsim statsbomb fit` 產生，請勿手動修改\n"
                    + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def typical_shot_context(shots) -> dict:
    """各距離的典型防守情境（線性）：角度內人數、最近防守者距離。作為「這次射門品質」的比較基準"""
    s = shots[shots.type == "Open Play"]
    d = np.hypot(105 - s.x.to_numpy(float), s.y.to_numpy(float) - 34)
    X = np.column_stack([np.ones(len(d)), d])
    cone = np.linalg.lstsq(X, s.n_cone.to_numpy(float), rcond=None)[0]
    near = np.linalg.lstsq(X, s.d_near.to_numpy(float), rcond=None)[0]
    return {"cone_a": round(float(cone[0]), 4), "cone_b": round(float(cone[1]), 4),
            "near_a": round(float(near[0]), 4), "near_b": round(float(near[1]), 4)}
