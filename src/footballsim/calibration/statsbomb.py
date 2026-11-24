"""StatsBomb 開放事件資料：下載、抽取（M4，設計文件 §6.2）

資料來源：StatsBomb Open Data（https://github.com/statsbomb/open-data），使用時須標註來源。
只使用 2015/16 五大聯賽「完整賽季」的資料；其他賽季多為單一球隊，會有偏差。

下載時逐場解析，只保留需要的欄位（data/raw/statsbomb/actions.csv.gz），不保存原始 JSON。
座標轉為本專案慣例：105 × 68 m、執行動作的隊伍進攻方向為 +x、y 越大越靠左路。
"""
from __future__ import annotations

import csv
import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
LEAGUES = {"EPL": (2, 27), "LaLiga": (11, 27), "Bundesliga": (9, 27), "SerieA": (12, 27),
           "Ligue1": (7, 27)}
DEFAULT_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb"
FIELDS = ["match_id", "league", "home", "period", "minute", "team", "type", "sub", "x", "y",
          "ex", "ey", "ok", "height", "cross", "through", "pressure", "xg", "shot_out", "body",
          "pattern", "possession", "idx"]
SHOT_FIELDS = ["match_id", "idx", "team", "type", "x", "y", "head", "n_cone", "d_near", "goal",
               "blocked", "on_target", "xg", "pattern", "possession"]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "footballsim-calibration"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _xy(loc):
    """StatsBomb 120×80 碼（y 向下）→ 105×68 m（y 越大越靠左）"""
    if not loc:
        return "", ""
    return round(loc[0] * 105 / 120, 2), round(68 - loc[1] * 68 / 80, 2)


def _shot_features(e: dict, match_id: int, team: int, x: float, y: float) -> list:
    """射門瞬間的位置特徵（freeze frame），與引擎共用 pitch.shot_context 的定義"""
    from ..engine.pitch import shot_context

    s = e["shot"]
    opp = [_xy(f["location"]) for f in s.get("freeze_frame", [])
           if not f.get("teammate") and f.get("position", {}).get("name") != "Goalkeeper"]
    n_cone, d_near, _ = shot_context(x, y, np.array(opp, float).reshape(-1, 2))
    out = s.get("outcome", {}).get("name", "")
    return [match_id, e["index"], team, s.get("type", {}).get("name", ""), x, y,
            int(s.get("body_part", {}).get("name") == "Head"), n_cone, round(d_near, 2),
            int(out == "Goal"), int(out == "Blocked"),
            int(out in ("Goal", "Saved", "Saved to Post")), s.get("statsbomb_xg", ""),
            e.get("play_pattern", {}).get("name", ""), e.get("possession", "")]


def parse_events(events: list, match_id: int, league: str, home_id: int,
                 shots_out: list | None = None) -> list[list]:
    rows = []
    for e in events:
        t = e["type"]["name"]
        team = e["team"]["id"]
        base = [match_id, league, int(team == home_id), e["period"], e["minute"], team]
        x, y = _xy(e.get("location"))
        press = int(bool(e.get("under_pressure")))
        pattern = e.get("play_pattern", {}).get("name", "")
        tail = [e.get("possession", ""), e.get("index", "")]
        n_before = len(rows)
        if t == "Pass":
            p = e["pass"]
            ex, ey = _xy(p.get("end_location"))
            out = p.get("outcome", {}).get("name", "")
            sub = p.get("type", {}).get("name", "Open Play")
            through = int(p.get("through_ball", False) or p.get("technique", {}).get("name") == "Through Ball")
            rows.append(base + ["pass", sub, x, y, ex, ey, int(out == ""),
                                p.get("height", {}).get("name", ""), int(p.get("cross", False)),
                                through, press, "", out, p.get("body_part", {}).get("name", ""), pattern])
        elif t == "Carry":
            ex, ey = _xy(e["carry"].get("end_location"))
            rows.append(base + ["carry", "", x, y, ex, ey, 1, "", 0, 0, press, "", "", "", pattern])
        elif t == "Dribble":
            ok = int(e["dribble"].get("outcome", {}).get("name") == "Complete")
            rows.append(base + ["take_on", "", x, y, "", "", ok, "", 0, 0, press, "", "", "", pattern])
        elif t == "Shot":
            s = e["shot"]
            ex, ey = _xy(s.get("end_location"))
            out = s.get("outcome", {}).get("name", "")
            rows.append(base + ["shot", s.get("type", {}).get("name", ""), x, y, ex, ey,
                                int(out == "Goal"), "", 0, 0, press, s.get("statsbomb_xg", ""), out,
                                s.get("body_part", {}).get("name", ""), pattern])
            if shots_out is not None and x != "":
                shots_out.append(_shot_features(e, match_id, team, x, y))
        elif t == "Foul Committed":
            card = e.get("foul_committed", {}).get("card", {}).get("name", "")
            rows.append(base + ["foul", card, x, y, "", "", 1, "", 0, 0, press, "", "", "", pattern])
        elif t == "Bad Behaviour":
            card = e.get("bad_behaviour", {}).get("card", {}).get("name", "")
            rows.append(base + ["card", card, x, y, "", "", 1, "", 0, 0, 0, "", "", "", pattern])
        elif t == "Offside":
            rows.append(base + ["offside", "", x, y, "", "", 1, "", 0, 0, 0, "", "", "", pattern])
        elif t == "Clearance":
            body = e.get("clearance", {}).get("body_part", {}).get("name", "")
            rows.append(base + ["clearance", "", x, y, "", "", 1, "", 0, 0, press, "", "", body, pattern])
        elif t in ("Ball Recovery", "Interception", "Goal Keeper"):
            # 僅用於判斷「之後誰拿到球」的序列分析
            rows.append(base + ["touch", t, x, y, "", "", 1, "", 0, 0, press, "", "", "", pattern])
        if len(rows) > n_before:
            rows[-1] = rows[-1] + tail
    return rows


def fetch(out_dir: Path = DEFAULT_DIR, per_league: int = 20, seed: int = 0, pause: float = 0.3,
          log=print) -> Path:
    """依序下載（單一連線、每次請求間隔 pause 秒），逐場抽取後寫入 actions.csv.gz"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "actions.csv.gz"
    rng = np.random.default_rng(seed)
    meta = []
    shots: list = []
    with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for league, (cid, sid) in LEAGUES.items():
            matches = json.loads(_get(f"{BASE}/matches/{cid}/{sid}.json"))
            pick = rng.choice(len(matches), size=min(per_league, len(matches)), replace=False)
            for n, i in enumerate(sorted(pick)):
                m = matches[int(i)]
                mid = m["match_id"]
                events = json.loads(_get(f"{BASE}/events/{mid}.json"))
                w.writerows(parse_events(events, mid, league, m["home_team"]["home_team_id"], shots))
                meta.append({"match_id": mid, "league": league,
                             "home_score": m["home_score"], "away_score": m["away_score"]})
                log(f"  {league} {n + 1}/{len(pick)}  match {mid}")
                time.sleep(pause)
    (out_dir / "matches.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with gzip.open(out_dir / "shots.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(SHOT_FIELDS)
        w.writerows(shots)
    return out


def load(path: Path = DEFAULT_DIR / "actions.csv.gz"):
    import pandas as pd

    return pd.read_csv(path)


def load_shots(path: Path = DEFAULT_DIR / "shots.csv.gz"):
    import pandas as pd

    return pd.read_csv(path)
