"""批次模擬、彙整與校準比對（§6.4）"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Optional

import numpy as np
import yaml

from ..config import CONFIG_DIR, load_params
from ..data.schema import League, MatchContext
from ..engine.match import simulate_match

TEAM_KEYS = ("goals", "shots", "sot", "blocked_shots", "xg", "passes", "passes_ok", "crosses",
             "crosses_ok", "corners", "fouls", "yellow", "red", "offsides", "penalties",
             "penalty_goals", "header_goals", "take_ons", "take_ons_ok", "subs", "injuries",
             "possession")
HOME_EDGE_POINTS = 1.0   # 判定賠率熱門時，主場約等於 1 點能力
DEFAULT_WORKERS = 1      # 低負荷預設：單一行程（曾兩度在批次模擬時導致主機當機）
MAX_WORKERS = 2
MEM_CAP_GB = float(os.environ.get("FOOTBALLSIM_MEM_GB", "3"))   # 每個行程的記憶體上限

_W: dict = {}


def limit_resources() -> None:
    """最低排程優先權，並設定記憶體硬上限：程式若失控，只會讓該行程出錯，不會拖垮整台電腦"""
    try:
        os.nice(19)
    except (AttributeError, OSError):
        pass
    try:
        import resource

        cap = int(MEM_CAP_GB * 1024 ** 3)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft == resource.RLIM_INFINITY or soft > cap:
            resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    except (ImportError, ValueError, OSError):
        pass


def _init(league_json: str, overrides: Optional[dict], engine: str = "event") -> None:
    limit_resources()
    _W["league"] = League.model_validate_json(league_json)
    ov = dict(overrides or {})
    sim_ov = {k[4:]: ov.pop(k) for k in list(ov) if k.startswith("sim.")}     # "sim." 開頭的覆寫給 v2 的 sim_params
    _W["params"] = load_params(overrides=ov or None)
    _W["engine"] = engine
    if engine == "tick":
        from ..sim.params import load_sim_params

        _W["sim_params"] = load_sim_params(overrides=sim_ov or None)


def _one(task) -> dict:
    h, a, seed = task
    lg = _W["league"]
    if _W.get("engine") == "tick":
        from ..sim import simulate_tick_match

        res = simulate_tick_match(lg.teams[h], lg.teams[a], params=_W["params"], sim_params=_W["sim_params"],
                                  seed=seed)
    else:
        res = simulate_match(lg.teams[h], lg.teams[a], params=_W["params"], seed=seed)
    return summarize(res, lg.teams[h].level, lg.teams[a].level)


SHOT_BINS = (0, 6, 11, 16.5, 25, 1000)
SHOT_BIN_NAMES = ("0-6", "6-11", "11-16.5", "16.5-25", "25+")


def summarize(res, level_home: float, level_away: float) -> dict:
    from ..engine import events as E

    st = res.stats()["teams"]
    dist = np.zeros(len(SHOT_BIN_NAMES))
    for e in res.events:
        if e.type in (E.SHOT, E.SHOT_FREEKICK):          # 不含十二碼，與 StatsBomb 目標一致
            x, y = (e.x, e.y) if e.team == 0 else (105 - e.x, 68 - e.y)
            d = float(np.hypot(105 - x, y - 34))
            dist[np.searchsorted(SHOT_BINS, d, side="right") - 1] += 1
    return {"score": res.score, "ht": res.ht_score, "levels": (level_home, level_away),
            "bip": res.ball_in_play, "stoppage": res.stoppage, "shot_dist": dist.tolist(),
            "teams": [{k: float(st[t].get(k, 0.0)) for k in TEAM_KEYS} for t in (0, 1)]}


def season_tasks(n_teams: int, n_matches: int, seed: int = 0) -> List[tuple]:
    """以完整主客雙循環為單位產生對戰（模擬真實聯賽的對戰組合）"""
    pairs = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
    rng = np.random.default_rng(seed)
    tasks = []
    while len(tasks) < n_matches:
        for i in rng.permutation(len(pairs)):
            tasks.append((*pairs[i], int(seed * 1_000_003 + len(tasks))))
            if len(tasks) >= n_matches:
                break
    return tasks


def run_batch(league: League, n_matches: int, workers: Optional[int] = None, seed: int = 0,
              overrides: Optional[dict] = None, tasks: Optional[Iterable] = None, engine: str = "event") -> List[dict]:
    tasks = list(tasks) if tasks is not None else season_tasks(len(league.teams), n_matches, seed)
    workers = min(workers or DEFAULT_WORKERS, MAX_WORKERS)
    js = league.model_dump_json()
    if workers == 1:
        _init(js, overrides, engine)
        return [_one(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(js, overrides, engine)) as ex:
        return list(ex.map(_one, tasks, chunksize=8))


def aggregate(S: List[dict]) -> dict:
    hg = np.array([s["score"][0] for s in S], float)
    ag = np.array([s["score"][1] for s in S], float)
    hh = np.array([s["ht"][0] for s in S], float)
    ah = np.array([s["ht"][1] for s in S], float)
    g, ht = hg + ag, hh + ah

    def tot(key: str, t: Optional[int] = None) -> np.ndarray:
        if t is None:
            return np.array([s["teams"][0][key] + s["teams"][1][key] for s in S], float)
        return np.array([s["teams"][t][key] for s in S], float)

    m: dict = {"matches": len(S)}
    m["goals_pm"] = g.mean()
    m["goals_var_over_mean"] = g.var(ddof=1) / max(g.mean(), 1e-9)
    m["home_goals"], m["away_goals"] = hg.mean(), ag.mean()
    m["H%"] = (hg > ag).mean() * 100
    m["D%"] = (hg == ag).mean() * 100
    m["A%"] = (hg < ag).mean() * 100
    m["0-0%"] = ((hg == 0) & (ag == 0)).mean() * 100
    m["corr_home_away"] = float(np.corrcoef(hg, ag)[0, 1])
    m["2H_goal_share%"] = (g - ht).sum() / max(g.sum(), 1) * 100
    shots, sot = tot("shots"), tot("sot")
    m["shots_pt"] = shots.mean() / 2
    m["sot_pt"] = sot.mean() / 2
    m["sot_ratio%"] = sot.sum() / shots.sum() * 100
    m["goals_per_shot%"] = g.sum() / shots.sum() * 100
    m["goals_per_sot%"] = g.sum() / sot.sum() * 100
    m["home_shots_share%"] = tot("shots", 0).sum() / shots.sum() * 100
    m["fouls_pt"] = tot("fouls").mean() / 2
    m["corners_pt"] = tot("corners").mean() / 2
    m["yellow_pt"] = tot("yellow").mean() / 2
    m["red_pm"] = tot("red").mean()
    m["home_yellow_share%"] = tot("yellow", 0).sum() / max(tot("yellow").sum(), 1) * 100
    lead_h, lead_a = hh > ah, ah > hh
    led = lead_h | lead_a
    win = (lead_h & (hg > ag)) | (lead_a & (ag > hg))
    m["HT_leader_win%"] = win[led].mean() * 100 if led.any() else float("nan")
    m["HT_leader_draw%"] = (hg == ag)[led].mean() * 100 if led.any() else float("nan")
    lv = np.array([s["levels"] for s in S], float)
    home_fav = lv[:, 0] + HOME_EDGE_POINTS >= lv[:, 1]
    m["fav_win%"] = np.where(home_fav, hg > ag, ag > hg).mean() * 100
    dist = np.bincount(np.minimum(g.astype(int), 7), minlength=8) / len(g) * 100
    m["total_goals_dist%"] = {i: round(float(v), 2) for i, v in enumerate(dist)}
    # 事件層
    passes = tot("passes")
    m["passes_pt"] = passes.mean() / 2
    m["pass_pct"] = tot("passes_ok").sum() / passes.sum() * 100
    m["crosses_pt"] = tot("crosses").mean() / 2
    m["cross_pct"] = tot("crosses_ok").sum() / max(tot("crosses").sum(), 1) * 100
    m["take_ons_pt"] = tot("take_ons").mean() / 2
    m["take_on_pct"] = tot("take_ons_ok").sum() / max(tot("take_ons").sum(), 1) * 100
    m["blocked_pct"] = tot("blocked_shots").sum() / shots.sum() * 100
    m["xg_per_shot"] = tot("xg").sum() / shots.sum()
    m["penalties_pm"] = tot("penalties").mean()
    m["header_goal_share%"] = tot("header_goals").sum() / max(g.sum(), 1) * 100
    m["bip_min"] = np.mean([s["bip"] for s in S]) / 60
    m["offsides_pt"] = tot("offsides").mean() / 2
    m["subs_pt"] = tot("subs").mean() / 2
    m["injuries_pt"] = tot("injuries").mean() / 2
    st = np.array([s["stoppage"] for s in S], float)
    m["stoppage_1h"], m["stoppage_2h"] = st[:, 0].mean(), st[:, 1].mean()
    sd = np.sum([s.get("shot_dist", [0] * len(SHOT_BIN_NAMES)) for s in S], axis=0)
    for name, v in zip(SHOT_BIN_NAMES, sd / max(sd.sum(), 1) * 100):
        m[f"shot_{name}%"] = float(v)
    return m


def load_targets() -> tuple[dict, dict]:
    top = yaml.safe_load((CONFIG_DIR / "targets" / "top5.yaml").read_text(encoding="utf-8"))
    ev = yaml.safe_load((CONFIG_DIR / "targets" / "event_level.yaml").read_text(encoding="utf-8"))
    return top, ev


def compare(m: dict) -> List[tuple]:
    """回傳 (指標, 模擬值, 目標, 下限, 上限, 狀態)；狀態：✓ 在範圍內、~ 接近、✗ 偏離"""
    top, ev = load_targets()
    rows = []
    for k, (lo, hi) in top["top5_range"].items():
        if k in m:
            rows.append((k, m[k], top["top5"][k], lo, hi))
    for k, (lo, hi) in ev.items():
        if k in m:
            rows.append((k, m[k], (lo + hi) / 2, lo, hi))
    out = []
    for k, v, tgt, lo, hi in rows:
        span = max(hi - lo, abs(tgt) * 0.04, 1e-6)
        if lo <= v <= hi:
            status = "✓"
        elif lo - span <= v <= hi + span:
            status = "~"
        else:
            status = "✗"
        out.append((k, float(v), float(tgt), float(lo), float(hi), status))
    return out


def format_comparison(rows: List[tuple]) -> str:
    lines = [f"{'指標':24s}{'模擬':>10s}{'目標':>10s}{'範圍':>20s}  狀態"]
    for k, v, tgt, lo, hi, st in rows:
        lines.append(f"{k:24s}{v:10.3f}{tgt:10.3f}   [{lo:7.3f}, {hi:7.3f}]  {st}")
    n_ok = sum(r[5] == "✓" for r in rows)
    n_near = sum(r[5] == "~" for r in rows)
    lines.append(f"\n✓ {n_ok}  ~ {n_near}  ✗ {len(rows) - n_ok - n_near}  （共 {len(rows)} 項）")
    return "\n".join(lines)
