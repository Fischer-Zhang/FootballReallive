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
    gk = {p["id"] for p in meta["players"] if p["player_role"]["acronym"] == "GK"}   # 追蹤的 player_id 是 match.json 的 id
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


# ------------------------------------------------------------------ 接球後的壓力（SkillCorner 事件＋追蹤）
RECEPTION_OFFSETS = (0.0, 0.5, 1.0, 1.5, 2.0)      # 接球後經過的秒數
ZONES = ((0.0, 35.0), (35.0, 70.0), (70.0, 105.0))


def reception_pressure(match_id: int) -> list[tuple]:
    """每次傳球接球（player_possession，start_type = pass_reception）：接球瞬間與之後 0.5、1、1.5、2 秒，
    持球者離最近對手（含門將）的距離，只在仍持球、且持球者與最近的對手都被偵測到時計。
    回傳 (進攻座標下的 x [0–105]、偏移序號、最近對手距離)。x 軸以持球隊進攻方向為正。"""
    import csv

    d = RAW / "skillcorner" / str(match_id)
    meta = json.loads((d / f"{match_id}_match.json").read_text(encoding="utf-8"))
    tr_of = {p["id"]: p["id"] for p in meta["players"]}           # 事件與追蹤都用 match.json 的球員 id
    team_of_tr = {p["id"]: p["team_id"] for p in meta["players"]}
    fr0, frames = None, {}
    with open(d / f"{match_id}_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("period") is None or not o.get("player_data"):
                continue
            frames[o["frame"]] = [(p["player_id"], p["x"], p["y"], bool(p.get("is_detected"))) for p in o["player_data"]]
    out = []
    with open(d / f"{match_id}_dynamic_events.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["event_type"] != "player_possession" or r["start_type"] != "pass_reception":
                continue
            tr = tr_of.get(int(r["player_id"])) if r["player_id"] else None
            if tr is None:
                continue
            f0, f1 = int(r["frame_start"]), int(r["frame_end"])
            team = team_of_tr.get(tr)
            side = 1.0 if r["attacking_side"] == "left_to_right" else -1.0
            x_att = float(r["x_start"]) * side + LENGTH / 2
            for k, off in enumerate(RECEPTION_OFFSETS):
                f = f0 + int(round(off * 10))
                if f > f1 or f not in frames:
                    break
                rows = frames[f]
                me = next((q for q in rows if q[0] == tr), None)
                if me is None or not me[3]:
                    break
                opp = [q for q in rows if team_of_tr.get(q[0]) not in (None, team) and q[3]]
                if not opp:
                    continue
                dq = [(((q[1] - me[1]) ** 2 + (q[2] - me[2]) ** 2) ** 0.5, q[0]) for q in opp]
                dmin, oid = min(dq)
                # 最近對手當下的速度（前後各 0.2 秒的中心差分；兩端都偵測到才算）
                v = np.nan
                a_, b_ = frames.get(f - 2), frames.get(f + 2)
                if a_ is not None and b_ is not None:
                    pa = next((q for q in a_ if q[0] == oid and q[3]), None)
                    pb = next((q for q in b_ if q[0] == oid and q[3]), None)
                    if pa and pb:
                        v = ((pb[1] - pa[1]) ** 2 + (pb[2] - pa[2]) ** 2) ** 0.5 / 0.4
                vc = np.nan                                            # 持球者自己的速度
                if a_ is not None and b_ is not None:
                    ca = next((q for q in a_ if q[0] == tr and q[3]), None)
                    cb = next((q for q in b_ if q[0] == tr and q[3]), None)
                    if ca and cb:
                        vc = ((cb[1] - ca[1]) ** 2 + (cb[2] - ca[2]) ** 2) ** 0.5 / 0.4
                out.append((x_att, k, float(min(dmin, 30.0)), float(v), float(vc)))
    return out


def format_reception(rows: list[tuple], name: str) -> str:
    a = np.array(rows, float)
    lines = [f"{name}：接球後最近對手距離（m；中位 [p25–p75]，n）／最近對手速度中位（m/s）——接球瞬間／0.5／1.0／1.5／2.0 秒"]
    for lo, hi in ZONES:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        cells = []
        for k in range(len(RECEPTION_OFFSETS)):
            v = a[m & (a[:, 1] == k), 2]
            sp = a[m & (a[:, 1] == k), 3]
            sp = sp[np.isfinite(sp)]
            cells.append(f"{np.median(v):.1f} [{np.percentile(v, 25):.1f}–{np.percentile(v, 75):.1f}] v{np.median(sp) if len(sp) else float('nan'):.1f} n={len(v)}" if len(v) > 20 else "—")
        lines.append(f"  x {lo:.0f}–{hi:.0f}：" + "｜".join(cells))
    return "\n".join(lines)


def engagement_motion(match_id: int, subtypes=("pressing", "pressure", "counter_press", "recovery_press")) -> list[tuple]:
    """逼搶事件（on_ball_engagement）期間防守者的速度與加速度（0.5 秒平滑、只用偵測到的位置）。
    回傳 (事件子類型, 開始後經過秒數, 速度 m/s, 切線加速度 m/s²) 的樣本。"""
    import csv

    d = RAW / "skillcorner" / str(match_id)
    traj = {}
    with open(d / f"{match_id}_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("period") is None or not o.get("player_data"):
                continue
            for p in o["player_data"]:
                if p.get("is_detected"):
                    traj.setdefault(p["player_id"], {})[o["frame"]] = (p["x"], p["y"])
    out = []
    with open(d / f"{match_id}_dynamic_events.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["event_type"] != "on_ball_engagement" or r["event_subtype"] not in subtypes or not r["player_id"]:
                continue
            tr = traj.get(int(r["player_id"]))
            if tr is None:
                continue
            f0, f1 = int(r["frame_start"]) - 10, int(r["frame_end"]) + 5
            fr = [f for f in range(f0, f1 + 1) if f in tr]
            if len(fr) < f1 - f0 - 2:
                continue
            p = np.array([tr[f] for f in range(f0, f1 + 1) if f in tr], float)
            if len(p) < 25:
                continue
            v, a = speed_acc(p, 10.0)
            for i in range(len(p)):
                if np.isfinite(v[i]) and np.isfinite(a[i]):
                    out.append((r["event_subtype"], (f0 + i - int(r["frame_start"])) / 10.0, float(v[i]), float(a[i])))
    return out


# ------------------------------------------------------------------ 防守方的陣型深度（SkillCorner 追蹤）
def defence_shape(match_id: int, every: int = 10) -> list[tuple]:
    """持球隊持球時（球被偵測到），防守方外野球員相對球的位置（座標以持球隊進攻方向為正）。
    每 every 個畫面取一次。回傳 (球 x, 球離中線的距離, 在球與球門之間的人數, 最深防守者的 x, 最深 4 人的平均 x)。
    防守方的人用含外推的追蹤位置（陣型深度要全員，被攝影機拍到的只是一部分）。"""
    d = RAW / "skillcorner" / str(match_id)
    meta = json.loads((d / f"{match_id}_match.json").read_text(encoding="utf-8"))
    team_of = {p["id"]: p["team_id"] for p in meta["players"]}
    gk = {p["id"] for p in meta["players"] if p["player_role"]["acronym"] == "GK"}
    home, away = meta["home_team"]["id"], meta["away_team"]["id"]
    rows_f = []
    gk_x = {}                                           # (period, team_id) → 門將 x 的累計，用來判斷進攻方向
    with open(d / f"{match_id}_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("period") is None or not o.get("player_data"):
                continue
            pd_ = o["player_data"]
            for p in pd_:
                if p["player_id"] in gk:
                    a = gk_x.setdefault((o["period"], team_of[p["player_id"]]), [0.0, 0])
                    a[0] += p["x"]
                    a[1] += 1
            if o["frame"] % every == 0 and o["possession"]["group"] and o["ball_data"]["is_detected"]:
                rows_f.append((o["period"], o["possession"]["group"], o["ball_data"]["x"], o["ball_data"]["y"],
                               [(p["player_id"], p["x"], p["y"]) for p in pd_]))
    out = []
    for period, group, bx, by, pls in rows_f:
        team = home if group == "home team" else away
        gx = gk_x.get((period, team))
        if not gx or not gx[1]:
            continue
        s = 1.0 if gx[0] / gx[1] < 0 else -1.0           # 門將在左邊 → 該隊往 +x 進攻
        xb = bx * s + LENGTH / 2
        defx = sorted(((p[1] * s + LENGTH / 2) for p in pls if team_of.get(p[0]) not in (None, team) and p[0] not in gk),
                      reverse=True)
        if len(defx) < 8:
            continue
        out.append((xb, abs(by), float(sum(1 for x in defx if x > xb)), float(defx[0]), float(np.mean(defx[:4]))))
    return out


def format_defence(rows: list[tuple], name: str) -> str:
    a = np.array(rows, float)
    lines = [f"{name}：球在持球隊進攻方向的 x 分段 → 球與球門之間的防守者人數 中位 [p25–p75]｜最深 4 人的平均 x − 球 x｜最深防守者 x − 球 x"]
    for lo in range(30, 100, 10):
        m = (a[:, 0] >= lo) & (a[:, 0] < lo + 10)
        if m.sum() < 30:
            continue
        n = a[m, 2]
        lines.append(f"  球 x {lo}–{lo + 10}：{np.median(n):.0f} [{np.percentile(n, 25):.0f}–{np.percentile(n, 75):.0f}]｜"
                     f"{np.median(a[m, 4] - a[m, 0]):+.1f} m｜{np.median(a[m, 3] - a[m, 0]):+.1f} m  (n={m.sum()})")
    return "\n".join(lines)


def attack_lateral(match_id: int, every: int = 10, x_lo: float = 50.0, x_hi: float = 75.0) -> list[tuple]:
    """持球隊持球時（球在進攻方向 x_lo–x_hi m），球前方 8 m 以上的攻方外野隊友離中線的距離與相對球的橫向位置（含外推位置）。
    回傳 (離中線距離, 有號橫向位置（球所在側為正）, 球離中線的距離)。"""
    d = RAW / "skillcorner" / str(match_id)
    meta = json.loads((d / f"{match_id}_match.json").read_text(encoding="utf-8"))
    team_of = {p["id"]: p["team_id"] for p in meta["players"]}
    gk = {p["id"] for p in meta["players"] if p["player_role"]["acronym"] == "GK"}
    home, away = meta["home_team"]["id"], meta["away_team"]["id"]
    gk_x, frames = {}, []
    with open(d / f"{match_id}_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if o.get("period") is None or not o.get("player_data"):
                continue
            for p in o["player_data"]:
                if p["player_id"] in gk:
                    a = gk_x.setdefault((o["period"], team_of[p["player_id"]]), [0.0, 0])
                    a[0] += p["x"]
                    a[1] += 1
            if o["frame"] % every == 0 and o["possession"]["group"] and o["ball_data"]["is_detected"]:
                frames.append((o["period"], o["possession"]["group"], o["ball_data"]["x"], o["ball_data"]["y"],
                               o["possession"]["player_id"], [(p["player_id"], p["x"], p["y"]) for p in o["player_data"]]))
    out = []
    for period, group, bx, by, holder, pls in frames:
        team = home if group == "home team" else away
        g = gk_x.get((period, team))
        if not g or not g[1]:
            continue
        s = 1.0 if g[0] / g[1] < 0 else -1.0
        xb = bx * s + LENGTH / 2
        if not x_lo <= xb < x_hi:
            continue
        sg = 1.0 if by >= 0 else -1.0                      # 追蹤座標的 y 以球場中線為 0
        for pid, px, py in pls:
            if team_of.get(pid) == team and pid not in gk and pid != holder and (px * s + LENGTH / 2) - xb >= 8.0:
                out.append((abs(py), py * sg, abs(by)))
    return out
