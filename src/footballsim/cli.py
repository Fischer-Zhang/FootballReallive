"""命令列介面：python -m footballsim <指令>

所有批次指令預設為低負荷（1 個行程、最低排程優先權、每行程記憶體上限 3 GB）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_LEAGUE = Path("data/league.json")


def _league(path: str | None, seed: int = 1):
    from .data.generator import generate_league
    from .data.repository import load_league

    if path and Path(path).exists():
        return load_league(path)
    if path:
        print(f"找不到 {path}，改用種子 {seed} 生成預設聯賽", file=sys.stderr)
    return generate_league(seed=seed)


def _overrides(items):
    out = {}
    for kv in items or []:
        k, v = kv.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def cmd_league(a) -> None:
    from .data.generator import generate_league
    from .data.repository import save_league

    lg = generate_league(seed=a.seed, n_teams=a.teams, spread=a.spread)
    path = save_league(lg, a.out)
    print(f"已生成 {len(lg.teams)} 隊（每隊 {len(lg.teams[0].players)} 人）→ {path}")
    for i, t in enumerate(lg.teams):
        print(f"  [{i:2d}] {t.name:6s} 水準 {t.level:5.1f}  {t.tactics.formation}")


def cmd_match(a) -> None:
    from .analytics.report import format_match
    from .config import load_params
    from .engine import simulate_match

    lg = _league(a.league)
    params = load_params(overrides=_overrides(a.set)) if a.set else None
    t = time.time()
    res = simulate_match(lg.teams[a.home], lg.teams[a.away], params=params, seed=a.seed)
    print(format_match(res, show_events=a.events))
    print(f"\n（模擬耗時 {time.time() - t:.2f} 秒，事件 {len(res.events)} 筆）")


def cmd_tickmatch(a) -> None:
    """v2 逐幀引擎：模擬一場比賽並輸出報告（含跑動距離）"""
    from .analytics.report import format_match
    from .sim import simulate_tick_match

    lg = _league(a.league)
    t = time.time()
    res = simulate_tick_match(lg.teams[a.home], lg.teams[a.away], seed=a.seed)
    print(format_match(res, show_events=a.events))
    d = sorted(res.distance.values(), reverse=True)
    print(f"\n跑動距離（km）：平均 {sum(d) / len(d) / 1000:.1f}，最高 {d[0] / 1000:.1f}")
    print(f"（逐幀模擬耗時 {time.time() - t:.1f} 秒，事件 {len(res.events)} 筆）")


def cmd_calibrate(a) -> None:
    from .calibration.runner import aggregate, compare, format_comparison, run_batch

    lg = _league(a.league)
    t = time.time()
    S = run_batch(lg, a.matches, workers=a.workers, seed=a.seed, overrides=_overrides(a.set) or None,
                  engine=a.engine)
    print(f"{a.matches} 場，耗時 {time.time() - t:.0f} 秒（{a.workers} 個行程）")
    print(format_comparison(compare(aggregate(S))))


def cmd_sensitivity(a) -> None:
    from .calibration.sensitivity import effect, format_effect, format_mirror, mirror_effect

    lg = _league(a.league)
    groups = a.groups.split(",") if a.groups else None
    if a.mirror:
        r = mirror_effect(lg, a.attr, groups, focus=a.focus, delta=a.delta, n=a.matches,
                          workers=a.workers)
        print(format_mirror(r))
        return
    r = effect(lg, a.attr, groups, focus=a.focus, delta=a.delta, reps=a.reps, workers=a.workers)
    print(format_effect(r))


def cmd_targets(a) -> None:
    from .calibration import targets

    if a.action == "fetch":
        paths = targets.fetch()
        print(f"已下載 {len(paths)} 個檔案到 {paths[0].parent}")
    doc = targets.compute()
    t = doc["top5"]
    print(f"已寫入 config/targets/top5.yaml（{t['matches']} 場）："
          f"每場進球 {t['goals_pm']:.2f}、主/和/客 {t['H%']:.1f}/{t['D%']:.1f}/{t['A%']:.1f}%")


def cmd_statsbomb(a) -> None:
    from .calibration import fit, statsbomb as sb
    from .config import load_params

    if a.action == "fetch":
        path = sb.fetch(per_league=a.per_league)
        print(f"已抽取 → {path}")
        return
    if a.action == "fetch360":
        from .calibration import statsbomb360 as s3

        path = s3.fetch(per_comp=a.per_comp)
        print(f"已抽取 360 位置資料 → {path}")
        return
    if a.action == "fit360":
        from .calibration import fit360, statsbomb360 as s3

        frames = s3.load()
        pool = fit360.box_run_pool(frames)
        n_spots = sum(len(f[2]) for f in pool)
        print(f"禁區跑位組合 → {fit360.save_box_pool(pool)}（{len(pool)} 個畫面、{n_spots} 個站位）")
        lat = fit360.fit_att_lateral(frames)
        print(f"橫向站位（攻守雙方）→ {fit360.save_att_lateral(lat)}（{sum(sum(r) for r in lat['_frames'])} 個畫面）")
        geo = fit360.geo_actions(frames)
        sp, cp, kp = (fit360.fit_shot_propensity(geo), fit360.fit_cross_propensity(geo),
                      fit360.fit_clear_propensity(geo))
        pc = fit360.fit_pass_choice(frames)
        pss, phh = fit360.fit_pass_success(frames)
        gp = fit360.fit_carry_propensity(frames)
        cc = fit360.fit_carry_choice(frames)
        tq = fit360.fit_through_propensity(frames)
        tc = fit360.fit_through_choice(frames)
        print(f"行為與傳球模型（位置特徵）→ {fit360.save_fitted360(sp, cp, kp, pc, pss, phh, gp, cc, tq, tc)}")
        for name, m in (("射門", sp), ("傳中", cp), ("解圍", kp), ("帶球", gp), ("傳球對象", pc), ("傳球成功率", pss),
                        ("傳球高度", phh), ("帶球方向", cc), ("身後球", tq), ("身後球接球者", tc)):
            print(f"  {name}：" + "、".join(f"{k}={v}" for k, v in m.items()))
        return
    df = sb.load()
    grid, info = fit.expected_threat(df)
    print(f"xT 網格 → {fit.save_xt(grid, info)}（{info['moves']} 次移動、{info['shots']} 次射門）")
    print(f"事件層目標 → {fit.save_event_targets(fit.event_targets(df))}")
    xg = fit.fit_xg(df, current=load_params()._raw["xg"])
    print(f"xG 擬合（請手動寫入 engine_params.yaml 的 xg）：{xg['coef']}，"
          f"log-loss {xg['log_loss_fit']:.4f}（目前 {xg.get('log_loss_current', float('nan')):.4f}，"
          f"StatsBomb xG {xg['log_loss_statsbomb_xg']:.4f}）")
    shots = sb.load_shots()
    sm = fit.fit_shot_models(shots)
    path = fit.save_fitted(sm, fit.transition_values(df, grid), typical=fit.typical_shot_context(shots))
    print(f"射門情境模型與轉換價值 → {path}（log-loss {sm['log_loss']['model']:.4f}，"
          f"StatsBomb xG {sm['log_loss']['statsbomb_xg']:.4f}）；行為模型改由 fit360 擬合")
    pc = fit.pass_curves(df)
    print(f"傳球成功率 {pc['overall%']}%（受壓 {pc['under_pressure%']}%、未受壓 {pc['no_pressure%']}%）")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="footballsim", description="屬性驅動足球比賽模擬")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("league", help="生成虛構聯賽")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--teams", type=int, default=20)
    p.add_argument("--spread", type=float, default=3.5)
    p.add_argument("--out", default=str(DEFAULT_LEAGUE))
    p.set_defaults(func=cmd_league)

    p = sub.add_parser("match", help="模擬一場比賽並輸出報告")
    p.add_argument("--league", default=str(DEFAULT_LEAGUE))
    p.add_argument("--home", type=int, default=0)
    p.add_argument("--away", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--events", action="store_true", help="列出射門、紅黃牌、換人等事件")
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="覆寫參數，例如 shot.k_gk=0.25")
    p.set_defaults(func=cmd_match)

    p = sub.add_parser("tickmatch", help="v2 逐幀引擎：模擬一場比賽並輸出報告")
    p.add_argument("--league", default=str(DEFAULT_LEAGUE))
    p.add_argument("--home", type=int, default=0)
    p.add_argument("--away", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--events", action="store_true", help="列出射門、紅黃牌、換人等事件")
    p.set_defaults(func=cmd_tickmatch)

    p = sub.add_parser("calibrate", help="批次模擬並與真實聯賽統計比對")
    p.add_argument("--league", default=str(DEFAULT_LEAGUE))
    p.add_argument("--matches", type=int, default=300)
    p.add_argument("--workers", type=int, default=1, help="行程數（預設 1、上限 2；低負荷）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="覆寫參數；v2 的 sim_params 以 sim. 開頭，例如 sim.motion.a_max=3")
    p.add_argument("--engine", choices=["event", "tick"], default="event", help="event = v1 事件驅動；tick = v2 逐幀")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("sensitivity", help="屬性敏感度分析（配對模擬）")
    p.add_argument("attr", help="屬性鍵名，例如 finishing；* 代表所有非隱藏屬性")
    p.add_argument("--groups", help="位置群組，逗號分隔：GK,CB,FB,DM,CM,AM,W,ST（預設全隊）")
    p.add_argument("--league", default=str(DEFAULT_LEAGUE))
    p.add_argument("--focus", type=int, default=None, help="受測球隊索引（預設中段球隊）")
    p.add_argument("--delta", type=int, default=15)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--mirror", action="store_true", help="鏡像對戰：加強版 vs 原版（建議，無共用基準偏差）")
    p.add_argument("--matches", type=int, default=160, help="鏡像對戰場數")
    p.set_defaults(func=cmd_sensitivity)

    p = sub.add_parser("targets", help="由 football-data.co.uk 計算校準目標")
    p.add_argument("action", choices=["fetch", "compute"])
    p.set_defaults(func=cmd_targets)

    p = sub.add_parser("statsbomb", help="StatsBomb 開放事件資料：下載抽取 / 擬合 xT、xG 與事件層目標")
    p.add_argument("action", choices=["fetch", "fetch360", "fit", "fit360"])
    p.add_argument("--per-league", type=int, default=20, help="每個聯賽抽樣場數（2015/16 五大聯賽）")
    p.add_argument("--per-comp", type=int, default=25, help="360：每項賽事抽樣場數（Euro 2024、WC 2022）")
    p.set_defaults(func=cmd_statsbomb)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
