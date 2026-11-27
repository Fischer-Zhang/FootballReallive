"""連續追蹤資料（Metrica 範例資料 25 Hz；SkillCorner 開放資料 10 Hz）的移動統計，用來校準 v2 的移動模型。

真實與模擬用同一套統計：位置先做 0.5 秒的移動平均，速度取 0.5 秒的中心差分，加速度取速度的 0.5 秒差分
（追蹤資料有雜訊，不平滑的話加速度會失真）。只看外野球員（門將另計）。
原始資料放在 data/raw/（不提交）；這個模組只讀取、不執行任何來自資料的內容。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
LENGTH, WIDTH = 105.0, 68.0
SPEED_ZONES = (0.7, 2.0, 4.0, 5.5, 7.0)            # 站立／走／慢跑／跑／高速／衝刺的邊界（m/s）
ZONE_NAMES = ("站立", "走", "慢跑", "跑", "高速", "衝刺")
ACC_BINS = (0.0, 2.0, 4.0, 6.0, 99.0)


# ------------------------------------------------------------------ 平滑與導數
def smooth_pos(p: np.ndarray, fps: float, window: float = 0.5) -> np.ndarray:
    """(T, 2) 位置的置中移動平均（NaN 傳播）"""
    k = max(int(round(window * fps)), 1)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    q = np.pad(p, ((pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(np.insert(q, 0, 0.0, axis=0), axis=0)
    return (c[k:] - c[:-k]) / k


def speed_acc(p: np.ndarray, fps: float, window: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(T, 2) 連續位置 → 速率 (T,) 與切線加速度 (T,)（m/s、m/s²）；兩端為 NaN"""
    s = smooth_pos(p, fps, window)
    h = max(int(round(window * fps / 2)), 1)
    v = np.full(len(s), np.nan)
    v[h:-h] = np.hypot(*(s[2 * h:] - s[:-2 * h]).T) / (2 * h / fps)
    a = np.full(len(s), np.nan)
    a[h:-h] = (v[2 * h:] - v[:-2 * h]) / (2 * h / fps) if len(v) > 4 * h else np.nan
    return v, a


def segments(valid: np.ndarray, min_len: int):
    """布林陣列中連續為 True 的區段 (start, end)（至少 min_len 長）"""
    out, start = [], None
    for i, ok in enumerate(np.append(valid, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


# ------------------------------------------------------------------ 統計
def motion_stats(trajs: list[np.ndarray], fps: float) -> dict:
    """trajs：每名外野球員一條 (T, 2) 軌跡（公尺，NaN = 沒有資料）。回傳速度分布、區間占比、加速度包絡、個人最高速、跑動距離"""
    sp_all, acc_all, per_max, dist, minutes = [], [], [], [], []
    for p in trajs:
        ok = np.isfinite(p).all(1)
        v_list, a_list = [], []
        for a0, a1 in segments(ok, int(2 * fps)):
            v, a = speed_acc(p[a0:a1], fps)
            v_list.append(v[np.isfinite(v)])
            a_list.append(np.column_stack([v[:-1], np.diff(v) * fps])[np.isfinite(np.diff(v))])
        if not v_list:
            continue
        v = np.concatenate(v_list)
        if len(v) < 60 * fps:
            continue
        sp_all.append(v)
        acc_all.append(np.vstack(a_list))
        per_max.append(float(np.percentile(v, 99.9)))
        dist.append(float(v.sum() / fps))
        minutes.append(len(v) / fps / 60.0)
    sp = np.concatenate(sp_all)
    ac = np.vstack(acc_all)
    edges = np.array(SPEED_ZONES)
    zone_share = np.bincount(np.searchsorted(edges, sp), minlength=len(edges) + 1) / len(sp)
    env = {}
    for lo, hi in zip(ACC_BINS[:-1], ACC_BINS[1:]):
        m = (ac[:, 0] >= lo) & (ac[:, 0] < hi)
        if m.sum() > 200:
            env[f"{lo:g}-{hi:g}"] = (float(np.percentile(ac[m, 1], 1)), float(np.percentile(ac[m, 1], 50)),
                                      float(np.percentile(ac[m, 1], 99)), int(m.sum()))
    return {"speed_pct": np.percentile(sp, [50, 90, 99, 99.9]).tolist(), "mean_speed": float(sp.mean()),
            "zone_share": zone_share.tolist(), "acc_env": env,
            "player_top": (float(np.mean(per_max)), float(np.std(per_max)), len(per_max)),
            "km_per_90": float(np.mean(np.array(dist) / np.array(minutes) * 90.0 / 1000.0)),
            "players": len(dist)}


def format_stats(name: str, s: dict) -> str:
    z = "  ".join(f"{n} {v * 100:4.1f}%" for n, v in zip(ZONE_NAMES, s["zone_share"]))
    env = "  ".join(f"[{k} m/s] {v[0]:+.1f}/{v[2]:+.1f}" for k, v in s["acc_env"].items())
    p = s["speed_pct"]
    return (f"{name}（{s['players']} 人）\n"
            f"  速度 p50/p90/p99/p99.9 = {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}/{p[3]:.2f} m/s；平均 {s['mean_speed']:.2f} m/s；"
            f"約 {s['km_per_90']:.1f} km／90 分\n  區間占比：{z}\n  個人最高速（p99.9）平均 {s['player_top'][0]:.2f} ± {s['player_top'][1]:.2f} m/s\n"
            f"  加速度包絡 p1/p99：{env}")


# ------------------------------------------------------------------ 載入（Metrica、SkillCorner）
def _metrica_csv(path: Path) -> tuple[np.ndarray, list]:
    """Metrica 範例資料 1、2（CSV，25 Hz，座標 0–1）→ (T, n_players, 2) 公尺，欄位名"""
    import pandas as pd

    df = pd.read_csv(path, skiprows=3, header=None)
    vals = df.iloc[:, 3:-2].to_numpy(float)                 # 去掉 Period、Frame、Time 與球的兩欄
    n = vals.shape[1] // 2
    arr = vals.reshape(len(df), n, 2) * (LENGTH, WIDTH)
    return arr, list(range(n))


def _is_gk(traj: np.ndarray) -> bool:
    ok = np.isfinite(traj).all(1)
    if ok.sum() < 1000:
        return False
    x = traj[ok, 0]
    return float(np.mean(np.minimum(x, LENGTH - x))) < 14.0


def metrica_trajs(game: int) -> list[np.ndarray]:
    d = RAW / "metrica" / f"Sample_Game_{game}"
    out = []
    for side in ("Home", "Away"):
        arr, _ = _metrica_csv(d / f"Sample_Game_{game}_RawTrackingData_{side}_Team.csv")
        for j in range(arr.shape[1]):
            t = arr[:, j, :]
            if np.isfinite(t).all(1).sum() > 1000 and not _is_gk(t):
                out.append(t)
    return out


def skillcorner_trajs(match_id: int, detected_only: bool = False) -> list[np.ndarray]:
    """SkillCorner 追蹤（10 Hz，公尺，原點在球場中心）→ 外野球員軌跡（以 0–105 × 0–68 表示）。
    detected_only：只用被攝影機看到的位置（外推的位置視為缺值）"""
    d = RAW / "skillcorner" / str(match_id)
    meta = json.loads((d / f"{match_id}_match.json").read_text(encoding="utf-8"))
    gk = {p["trackable_object"] for p in meta["players"] if p["player_role"]["acronym"] == "GK"}   # 追蹤的 player_id 是 trackable_object
    frames, pid_pos = [], {}
    with open(d / f"{match_id}_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("period") is None or not o.get("player_data"):
                continue
            fi = o["frame"]
            for p in o["player_data"]:
                pid_pos.setdefault(p["player_id"], []).append((fi, p["x"], p["y"], bool(p.get("is_detected"))))
            frames.append(fi)
    out = []
    for pid, rows in pid_pos.items():
        if pid in gk:
            continue
        a = np.array(rows, float)
        full = np.full((int(a[-1, 0] - a[0, 0]) + 1, 2), np.nan)
        idx = (a[:, 0] - a[0, 0]).astype(int)
        full[idx] = a[:, 1:3] + (LENGTH / 2, WIDTH / 2)
        if detected_only:
            det = np.zeros(len(full), bool)
            det[idx] = a[:, 3] > 0
            full[~det] = np.nan
        if np.isfinite(full).all(1).sum() > 6000 and not _is_gk(full):
            out.append(full)
    return out
