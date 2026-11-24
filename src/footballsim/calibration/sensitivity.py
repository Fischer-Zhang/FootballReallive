"""屬性敏感度分析（§6.5）：證明屬性真的影響比賽結果

做法：選一支球隊，對聯賽其他每支球隊踢主客場（可重複數輪），先以原始陣容模擬，
再把指定位置群組球員的某屬性 +delta，以「相同的隨機種子」重跑（共同隨機數），比較差異。
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ..data.schema import League
from ..model.attributes import GROUP_OF, INDEX
from ..model.positions import GROUP
from .runner import run_batch


def boost(league: League, team_idx: int, attr: str, groups: Optional[Iterable[str]], delta: int
          ) -> League:
    """回傳深拷貝：team_idx 隊中屬於 groups（位置群組，如 ST、W；None = 全隊）的球員 attr +delta。
    attr = "*" 代表所有非隱藏屬性（整體能力）。"""
    attrs = [a for a in INDEX if GROUP_OF[a] != "hidden"] if attr == "*" else [attr]
    for a in attrs:
        if a not in INDEX:
            raise KeyError(f"未知屬性：{a}")
    lg = league.model_copy(deep=True)
    groups = set(groups) if groups else None
    for p in lg.teams[team_idx].players:
        if groups is None or GROUP[p.position] in groups:
            for a in attrs:
                model = getattr(p, GROUP_OF[a])
                setattr(model, a, int(min(100, max(1, getattr(model, a) + delta))))
    return lg


def focus_tasks(n_teams: int, focus: int, reps: int, seed: int = 0) -> list:
    tasks = []
    for r in range(reps):
        for opp in range(n_teams):
            if opp == focus:
                continue
            tasks.append((focus, opp, seed * 7919 + r * 1000 + opp))
            tasks.append((opp, focus, seed * 7919 + r * 1000 + 500 + opp))
    return tasks


def _side_stats(tasks, S, focus) -> dict:
    rows = []
    for (h, _a, _s), s in zip(tasks, S):
        me = 0 if h == focus else 1
        op = 1 - me
        gf, ga = s["score"][me], s["score"][op]
        T, O = s["teams"][me], s["teams"][op]
        rows.append((gf, ga, 3 * (gf > ga) + (gf == ga), T["shots"], T["xg"], T["sot"],
                     T["passes_ok"], T["passes"], O["sot"], T["fouls"], T["take_ons_ok"],
                     T["take_ons"]))
    return np.array(rows, float)


def baseline(league: League, focus: Optional[int] = None, reps: int = 2, workers: int = 1,
             seed: int = 0) -> np.ndarray:
    """基準模擬（可在多個屬性之間共用，省下重複運算）"""
    n = len(league.teams)
    focus = n // 2 if focus is None else focus
    tasks = focus_tasks(n, focus, reps, seed)
    return _side_stats(tasks, run_batch(league, 0, workers=workers, tasks=tasks), focus)


def effect(league: League, attr: str, groups: Optional[Iterable[str]] = None, focus: Optional[int] = None,
           delta: int = 15, reps: int = 2, workers: int = 1, seed: int = 0,
           base: Optional[np.ndarray] = None) -> dict:
    n = len(league.teams)
    focus = n // 2 if focus is None else focus
    tasks = focus_tasks(n, focus, reps, seed)
    if base is None:
        base = baseline(league, focus, reps, workers, seed)
    boosted_league = boost(league, focus, attr, groups, delta)
    new = _side_stats(tasks, run_batch(boosted_league, 0, workers=workers, tasks=tasks), focus)

    def mean_diff(col: int):
        d = new[:, col] - base[:, col]
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))

    def ratio(num: int, den: int, arr) -> float:
        return float(arr[:, num].sum() / max(arr[:, den].sum(), 1e-9))

    gd_b, gd_n = base[:, 0] - base[:, 1], new[:, 0] - new[:, 1]
    d_gd = gd_n - gd_b
    return {
        "attr": attr, "groups": sorted(groups) if groups else "全隊", "delta": delta,
        "matches": len(tasks),
        "goal_diff": (float(d_gd.mean()), float(d_gd.std(ddof=1) / np.sqrt(len(d_gd)))),
        "goals_for": mean_diff(0), "goals_against": mean_diff(1), "points": mean_diff(2),
        "shots": mean_diff(3), "xg": mean_diff(4),
        "conversion%": (100 * ratio(0, 3, base), 100 * ratio(0, 3, new)),
        "xg_per_shot": (ratio(4, 3, base), ratio(4, 3, new)),
        "pass%": (100 * ratio(6, 7, base), 100 * ratio(6, 7, new)),
        "save%_vs": (100 * (1 - ratio(1, 8, base)), 100 * (1 - ratio(1, 8, new))),
        "take_on%": (100 * ratio(10, 11, base), 100 * ratio(10, 11, new)),
        "fouls": mean_diff(9),
    }


def mirror_effect(league: League, attr: str, groups: Optional[Iterable[str]] = None,
                  focus: Optional[int] = None, delta: int = 20, n: int = 160, workers: int = 1,
                  seed: int = 0) -> dict:
    """鏡像對戰：屬性加強版的球隊 vs 原版的自己，主客場輪流（抵銷主場優勢）。

    兩隊只差在指定屬性，沒有共用基準的運氣偏差；淨勝球的期望值就是屬性的效果。"""
    nt = len(league.teams)
    focus = nt // 2 if focus is None else focus
    lg = league.model_copy(deep=True)
    boosted = boost(league, focus, attr, groups, delta).teams[focus]
    boosted.team_id = boosted.team_id + "+"
    boosted.name = boosted.name + "+"
    for pl in boosted.players:               # 球員 ID 必須與原版不同（引擎以 ID 存放當日屬性）
        pl.player_id = pl.player_id + "+"
    lg.teams.append(boosted)
    b = nt
    tasks = [((b, focus) if i % 2 == 0 else (focus, b)) + (seed * 100_003 + i,) for i in range(n)]
    S = run_batch(lg, 0, workers=workers, tasks=tasks)
    gd, pts, conv_b, conv_o, sh_b, sh_o = [], [], [0.0, 0.0], [0.0, 0.0], [], []
    for (h, _a, _s), s in zip(tasks, S):
        me = 0 if h == b else 1
        g_me, g_op = s["score"][me], s["score"][1 - me]
        gd.append(g_me - g_op)
        pts.append(3 * (g_me > g_op) + (g_me == g_op))
        conv_b[0] += g_me
        conv_b[1] += s["teams"][me]["shots"]
        conv_o[0] += g_op
        conv_o[1] += s["teams"][1 - me]["shots"]
        sh_b.append(s["teams"][me]["shots"])
        sh_o.append(s["teams"][1 - me]["shots"])
    gd, pts = np.array(gd, float), np.array(pts, float)
    return {"attr": attr, "groups": sorted(groups) if groups else "全隊", "delta": delta, "matches": n,
            "goal_diff": (float(gd.mean()), float(gd.std(ddof=1) / np.sqrt(n))),
            "points": (float(pts.mean()), float(pts.std(ddof=1) / np.sqrt(n))),
            "win%": float((gd > 0).mean() * 100), "loss%": float((gd < 0).mean() * 100),
            "conversion%": (100 * conv_o[0] / max(conv_o[1], 1), 100 * conv_b[0] / max(conv_b[1], 1)),
            "shots": (float(np.mean(sh_o)), float(np.mean(sh_b)))}


def format_mirror(r: dict) -> str:
    return (f"{r['attr']} +{r['delta']}（{r['groups']}，加強版 vs 原版，{r['matches']} 場）：每場淨勝球 "
            f"{r['goal_diff'][0]:+.2f} ± {r['goal_diff'][1]:.2f}｜勝 {r['win%']:.0f}% 負 {r['loss%']:.0f}%"
            f"｜每場積分 {r['points'][0]:.2f}｜轉換率 原版 {r['conversion%'][0]:.1f}% → 加強版 "
            f"{r['conversion%'][1]:.1f}%｜射門 {r['shots'][0]:.1f} → {r['shots'][1]:.1f}")


def format_effect(r: dict) -> str:
    def pm(v):
        return f"{v[0]:+.2f} ± {v[1]:.2f}"

    def ab(v, fmt="{:.1f}"):
        return f"{fmt.format(v[0])} → {fmt.format(v[1])}"

    return (f"{r['attr']} +{r['delta']}（{r['groups']}，{r['matches']} 場配對）\n"
            f"  每場淨勝球 {pm(r['goal_diff'])}｜進球 {pm(r['goals_for'])}｜失球 {pm(r['goals_against'])}"
            f"｜積分 {pm(r['points'])}\n"
            f"  射門 {pm(r['shots'])}｜xG {pm(r['xg'])}｜轉換率 {ab(r['conversion%'])}%"
            f"｜每次射門 xG {ab(r['xg_per_shot'], '{:.3f}')}\n"
            f"  傳球成功率 {ab(r['pass%'])}%｜門將撲救率（對手射正）{ab(r['save%_vs'])}%"
            f"｜過人成功率 {ab(r['take_on%'])}%｜犯規 {pm(r['fouls'])}")
