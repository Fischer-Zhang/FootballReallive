"""由 football-data.co.uk 的 CSV 計算校準目標（設計文件 §6.1、附錄 C）

下載：python -m footballsim targets fetch
計算：python -m footballsim targets compute   → config/targets/top5.yaml
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from ..config import CONFIG_DIR, PROJECT_ROOT

SEASONS = ("2223", "2324", "2425", "2526")
TOP5 = ("E0", "SP1", "D1", "I1", "F1")
SECOND = ("E1",)
NAMES = {"E0": "EPL", "SP1": "LaLiga", "D1": "Bundesliga", "I1": "SerieA", "F1": "Ligue1",
         "E1": "Championship"}
COLS = ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
        "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A"]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "football_data"
URL = "https://football-data.co.uk/mmz4281/{season}/{div}.csv"


def fetch(out_dir: Path = DEFAULT_RAW, seasons=SEASONS, divisions=TOP5 + SECOND) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seasons:
        for d in divisions:
            path = out_dir / f"{s}_{d}.csv"
            req = urllib.request.Request(URL.format(season=s, div=d),
                                         headers={"User-Agent": "footballsim-calibration"})
            with urllib.request.urlopen(req, timeout=30) as resp:   # urllib 會自動跟隨轉址
                path.write_bytes(resp.read())
            paths.append(path)
    return paths


def _load(raw_dir: Path, div: str):
    import pandas as pd

    frames = []
    for f in sorted(Path(raw_dir).glob(f"*_{div}.csv")):
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1")
        frames.append(df[[c for c in COLS if c in df.columns]].dropna(subset=["FTHG", "FTAG"]))
    if not frames:
        raise FileNotFoundError(f"{raw_dir} 中找不到 *_{div}.csv，請先執行 targets fetch")
    return pd.concat(frames, ignore_index=True)


def summarize(df) -> dict:
    g, ht = df.FTHG + df.FTAG, df.HTHG + df.HTAG
    lead_h, lead_a = df.HTHG > df.HTAG, df.HTAG > df.HTHG
    led = lead_h | lead_a
    win = (lead_h & (df.FTR == "H")) | (lead_a & (df.FTR == "A"))
    o = df[["B365H", "B365D", "B365A"]].dropna()
    inv = 1 / o.values
    p = inv / inv.sum(axis=1, keepdims=True)
    res = df.loc[o.index, "FTR"].map({"H": 0, "D": 1, "A": 2}).values
    fav = np.where(p[:, 0] >= p[:, 2], 0, 2)
    pfav = p[np.arange(len(p)), fav]
    out = {
        "matches": int(len(df)),
        "goals_pm": g.mean(), "goals_var_over_mean": g.var() / g.mean(),
        "home_goals": df.FTHG.mean(), "away_goals": df.FTAG.mean(),
        "H%": (df.FTR == "H").mean() * 100, "D%": (df.FTR == "D").mean() * 100,
        "A%": (df.FTR == "A").mean() * 100,
        "0-0%": ((df.FTHG == 0) & (df.FTAG == 0)).mean() * 100,
        "corr_home_away": float(np.corrcoef(df.FTHG, df.FTAG)[0, 1]),
        "2H_goal_share%": (g - ht).sum() / g.sum() * 100,
        "shots_pt": (df.HS + df.AS).mean() / 2, "sot_pt": (df.HST + df.AST).mean() / 2,
        "sot_ratio%": (df.HST + df.AST).sum() / (df.HS + df.AS).sum() * 100,
        "goals_per_shot%": g.sum() / (df.HS + df.AS).sum() * 100,
        "goals_per_sot%": g.sum() / (df.HST + df.AST).sum() * 100,
        "home_shots_share%": df.HS.sum() / (df.HS + df.AS).sum() * 100,
        "fouls_pt": (df.HF + df.AF).mean() / 2, "corners_pt": (df.HC + df.AC).mean() / 2,
        "yellow_pt": (df.HY + df.AY).mean() / 2, "red_pm": (df.HR + df.AR).mean(),
        "home_yellow_share%": df.HY.sum() / (df.HY + df.AY).sum() * 100,
        "HT_leader_win%": win[led].mean() * 100,
        "HT_leader_draw%": (led & (df.FTR == "D"))[led].mean() * 100,
        "fav_win%": (res == fav).mean() * 100,
    }
    dist = (g.clip(upper=7).value_counts(normalize=True).sort_index() * 100)
    out["total_goals_dist%"] = {int(k): round(float(v), 2) for k, v in dist.items()}
    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]
    idx = np.digitize(pfav, bins) - 1
    grad = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum():
            grad.append({"implied": round(float(pfav[m].mean()) * 100, 1), "n": int(m.sum()),
                         "win%": round(float((res[m] == fav[m]).mean()) * 100, 1),
                         "draw%": round(float((res[m] == 1).mean()) * 100, 1)})
    out["favourite_gradient"] = grad
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def compute(raw_dir: Path = DEFAULT_RAW, out_path: Path | None = None) -> dict:
    import pandas as pd
    import yaml

    leagues = {d: _load(raw_dir, d) for d in TOP5 + SECOND}
    per = {NAMES[d]: summarize(leagues[d]) for d in TOP5 + SECOND}
    pooled = summarize(pd.concat([leagues[d] for d in TOP5], ignore_index=True))
    scalar = [k for k, v in pooled.items() if isinstance(v, float)]
    ranges = {k: [min(per[NAMES[d]][k] for d in TOP5), max(per[NAMES[d]][k] for d in TOP5)]
              for k in scalar}
    doc = {"source": "football-data.co.uk，" + "/".join(f"20{s[:2]}-{s[2:]}" for s in SEASONS)
                     + "，五大聯賽",
           "top5": pooled, "top5_range": ranges, "per_league": per}
    out_path = Path(out_path or CONFIG_DIR / "targets" / "top5.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return doc
