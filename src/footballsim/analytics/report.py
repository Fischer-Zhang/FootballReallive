"""終端機比賽報告（中文）"""
from __future__ import annotations

from ..engine import events as E

_ROWS = [("進球", "goals", "{:.0f}"), ("xG", "xg", "{:.2f}"), ("射門", "shots", "{:.0f}"),
         ("射正", "sot", "{:.0f}"), ("被封堵", "blocked_shots", "{:.0f}"),
         ("控球率 %", "possession", "{:.0f}"), ("傳球", "passes", "{:.0f}"),
         ("傳球成功率 %", "pass_pct", "{:.0f}"), ("傳中", "crosses", "{:.0f}"),
         ("過人（成功）", "take_ons_ok", "{:.0f}"), ("搶斷成功", "tackles_won", "{:.0f}"),
         ("攔截", "interceptions", "{:.0f}"), ("角球", "corners", "{:.0f}"),
         ("越位", "offsides", "{:.0f}"), ("犯規", "fouls", "{:.0f}"), ("黃牌", "yellow", "{:.0f}"),
         ("紅牌", "red", "{:.0f}"), ("撲救", "saves", "{:.0f}")]


def format_match(res, show_events: bool = False) -> str:
    st = res.stats()
    names = {pid: info["name"] for pid, info in res.players.items()}
    lines = [f"{res.home} {res.score[0]} : {res.score[1]} {res.away}"
             f"   （半場 {res.ht_score[0]}:{res.ht_score[1]}，補時 +{res.stoppage[0]} / +{res.stoppage[1]}）"]
    for g in st["goals"]:
        side = res.home if g["team"] == 0 else res.away
        how = {"shot_penalty": "（十二碼）", "shot_freekick": "（自由球）"}.get(g["type"], "")
        head = "（頭槌）" if g["body"] == "head" else ""
        ast = f"，助攻 {names.get(g['assist'], '')}" if g["assist"] else ""
        lines.append(f"  {g['minute']:5.1f}'  {side}  {names.get(g['player'], '?')}{how}{head}{ast}"
                     f"  xG {g['xg']:.2f}")
    lines.append("")
    w = max(len(res.home), len(res.away), 6)
    lines.append(f"{'':14s}{res.home:>{w}s}  {res.away:>{w}s}")
    for label, key, fmt in _ROWS:
        a = fmt.format(st["teams"][0].get(key, 0))
        b = fmt.format(st["teams"][1].get(key, 0))
        lines.append(f"{label:14s}{a:>{w}s}  {b:>{w}s}")
    lines.append(f"淨比賽時間 {res.ball_in_play / 60:.1f} 分鐘；換人 {res.subs[0]} / {res.subs[1]}")
    top = sorted(((p.get("rating", 0), pid) for pid, p in st["players"].items()
                  if res.minutes.get(pid, 0) > 0), reverse=True)[:5]
    lines.append("最佳球員：" + "、".join(f"{names.get(pid, pid)} {r:.1f}" for r, pid in top))
    if show_events:
        lines.append("")
        for ev in res.events:
            if ev.type in E.SHOT_TYPES or ev.type in (E.CARD, E.SUBSTITUTION, E.INJURY):
                who = names.get(ev.player, "")
                lines.append(f"  {ev.minute:5.1f}' [{ev.team}] {ev.type:14s} {ev.result:12s} {who}")
    return "\n".join(lines)
