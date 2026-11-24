"""StatsBomb 360 資料：事件瞬間「畫面內所有球員」的位置（M4 站位校準）

資料來源：StatsBomb Open Data 360（https://github.com/statsbomb/open-data），使用時須標註來源。
只用「每場都有 360」的完整賽事：UEFA Euro 2024、FIFA World Cup 2022，避免單一球隊偏差。
（俱樂部賽事的 360 只涵蓋單一球隊的比賽。）

下載後逐場抽取，不保存原始 JSON。抽取的事件（皆為運動戰）：
- cross  ：傳中
- pass   ：其他傳球
- carry  ：帶球（長度 ≥ 1 m），供帶球路線封堵模型擬合
- take_on：過人
- shot   ：射門（outcome 欄位為射門結果；shot 欄位為 StatsBomb xG）
- clear  ：解圍（height 欄位為身體部位）
射門、傳中、解圍的行為模型用「位置特徵」擬合（最近對手距離等），不用人工標記的 under_pressure。

每列含事件資訊、可視範圍多邊形（vis）與畫面內所有球員座標（frame，緊湊字串）：
  A=持球者  T=隊友  G=本隊門將  O=對手  K=對方門將；例如 "A98.1,12.3;T100.2,30.1;K104.0,33.9"
座標慣例同 statsbomb._xy：105 × 68 m、執行動作的隊伍進攻方向為 +x、y 越大越靠左。
"""
from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT
from .statsbomb import BASE, _get, _xy

COMPETITIONS = {"Euro2024": (55, 282), "WC2022": (43, 106)}
DEFAULT_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb360"
SET_PIECES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
FIELDS = ["comp", "match_id", "idx", "team", "minute", "kind", "x", "y", "ex", "ey", "ok", "outcome",
          "height", "pressure", "body", "pattern", "next", "shot", "vis", "frame"]
CARRY_MIN_LEN = 1.0
SHOT_WINDOW_S = 6.0


def _t(e: dict) -> float:
    h, m, s = e["timestamp"].split(":")
    return e["period"] * 10000 + int(h) * 3600 + int(m) * 60 + float(s)


def _frame(f: dict) -> str:
    out = []
    for p in f["freeze_frame"]:
        x, y = _xy(p["location"])
        if p.get("actor"):
            tag = "A"
        elif p.get("teammate"):
            tag = "G" if p.get("keeper") else "T"
        else:
            tag = "K" if p.get("keeper") else "O"
        out.append(f"{tag}{x:.1f},{y:.1f}")
    return ";".join(out)


def _vis(f: dict) -> str:
    v = f.get("visible_area") or []
    pts = [_xy((v[i], v[i + 1])) for i in range(0, len(v) - 1, 2)]
    return ";".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def _after(events: list, i: int, team: int) -> tuple[str, str]:
    """傳球之後的三個事件（略過壓迫），以及同隊在 6 秒內的第一次射門"""
    t0 = _t(events[i])
    nxt, shot = [], ""
    for e in events[i + 1: i + 12]:
        name = e["type"]["name"]
        if len(nxt) < 3 and name != "Pressure":
            sub = e.get("goalkeeper", {}).get("type", {}).get("name", "") if name == "Goal Keeper" else ""
            nxt.append(f"{name}{'/' + sub if sub else ''}:{int(e['team']['id'] == team)}")
        if not shot and name == "Shot" and e["team"]["id"] == team and _t(e) - t0 <= SHOT_WINDOW_S:
            s = e["shot"]
            sx, sy = _xy(e.get("location"))
            shot = (f"{sx},{sy},{s.get('body_part', {}).get('name', '')},{s.get('statsbomb_xg', '')},"
                    f"{s.get('outcome', {}).get('name', '')}")
        if _t(e) - t0 > SHOT_WINDOW_S:
            break
    return "|".join(nxt), shot


def parse_match(events: list, frames: list, comp: str, match_id: int) -> list[list]:
    fr = {f["event_uuid"]: f for f in frames}
    rows = []
    for i, e in enumerate(events):
        t = e["type"]["name"]
        if t not in ("Pass", "Carry", "Dribble", "Shot", "Clearance") or e["id"] not in fr:
            continue
        x, y = _xy(e.get("location"))
        if x == "":
            continue
        height = body = ""
        if t == "Pass":
            p = e["pass"]
            if p.get("type", {}).get("name", "Open Play") in SET_PIECES:
                continue
            kind = "cross" if p.get("cross") else "pass"
            ex, ey = _xy(p.get("end_location"))
            out = p.get("outcome", {}).get("name", "")
            height = p.get("height", {}).get("name", "")
            body = p.get("body_part", {}).get("name", "")
        elif t == "Carry":
            kind, out = "carry", ""
            ex, ey = _xy(e["carry"].get("end_location"))
            if ex == "" or ((ex - x) ** 2 + (ey - y) ** 2) ** 0.5 < CARRY_MIN_LEN:
                continue
        elif t == "Dribble":
            kind, ex, ey = "take_on", "", ""
            out = e["dribble"].get("outcome", {}).get("name", "")
            out = "" if out == "Complete" else out
        elif t == "Shot":
            sh = e["shot"]
            if sh.get("type", {}).get("name", "") != "Open Play":
                continue
            kind = "shot"
            ex, ey = _xy(sh.get("end_location"))
            out = sh.get("outcome", {}).get("name", "")
            out = "" if out == "Goal" else out
            body = sh.get("body_part", {}).get("name", "")
        else:
            kind, ex, ey, out = "clear", "", "", ""
            body = e.get("clearance", {}).get("body_part", {}).get("name", "")
        team = e["team"]["id"]
        nxt, shot = _after(events, i, team)
        if kind == "shot":
            shot = str(e["shot"].get("statsbomb_xg", ""))
        f = fr[e["id"]]
        rows.append([comp, match_id, e["index"], team, e["minute"], kind, x, y, ex, ey, int(out == ""), out,
                     height, int(bool(e.get("under_pressure"))), body,
                     e.get("play_pattern", {}).get("name", ""), nxt, shot, _vis(f), _frame(f)])
    return rows


def fetch(out_dir: Path = DEFAULT_DIR, per_comp: int = 25, seed: int = 0, pause: float = 0.5,
          log=print) -> Path:
    """依序下載（單一連線、請求間隔 pause 秒），逐場抽取後寫入 frames.csv.gz"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "frames.csv.gz"
    rng = np.random.default_rng(seed)
    meta = []
    with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for comp, (cid, sid) in COMPETITIONS.items():
            matches = [m for m in json.loads(_get(f"{BASE}/matches/{cid}/{sid}.json"))
                       if m.get("match_status_360") == "available"]
            pick = rng.choice(len(matches), size=min(per_comp, len(matches)), replace=False)
            for n, k in enumerate(sorted(pick)):
                m = matches[int(k)]
                mid = m["match_id"]
                events = json.loads(_get(f"{BASE}/events/{mid}.json"))
                time.sleep(pause)
                frames = json.loads(_get(f"{BASE}/three-sixty/{mid}.json"))
                rows = parse_match(events, frames, comp, mid)
                w.writerows(rows)
                meta.append({"match_id": mid, "comp": comp, "rows": len(rows)})
                log(f"  {comp} {n + 1}/{len(pick)}  match {mid}  rows {len(rows)}")
                del events, frames
                time.sleep(pause)
    (out_dir / "matches.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------ 讀取與幾何
def parse_frame(s: str) -> dict:
    """緊湊字串 → {"A": (1,2), "T": (n,2), "G": ..., "O": ..., "K": ...}"""
    out = {k: [] for k in "ATGOK"}
    if isinstance(s, str) and s:
        for tok in s.split(";"):
            a, b = tok[1:].split(",")
            out[tok[0]].append((float(a), float(b)))
    return {k: np.asarray(v, float).reshape(-1, 2) for k, v in out.items()}


def parse_poly(s: str) -> np.ndarray:
    if not isinstance(s, str) or not s:
        return np.zeros((0, 2))
    return np.asarray([[float(v) for v in tok.split(",")] for tok in s.split(";")], float)


def inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """點是否在可視多邊形內（射線法，向量化）"""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    cond = (y1 > y) != (y2 > y)
    xint = (x2 - x1) * (y - y1) / np.where(y2 == y1, 1e-9, y2 - y1) + x1
    return ((cond & (x < xint)).sum(1) % 2) == 1


def load(path: Path = DEFAULT_DIR / "frames.csv.gz"):
    import pandas as pd

    return pd.read_csv(path)
