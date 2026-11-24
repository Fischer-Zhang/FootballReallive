"""球員、球隊、聯賽生成器（設計文件 §3.13）

屬性 = 能力水準 L ＋ 位置原型偏移 ＋ 群組相關雜訊 ＋ 個別雜訊 ＋ 年齡效果 ＋ 身材效果。
所有隨機性都來自傳入的 numpy Generator，因此同一個種子會生成同一個聯賽。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from ..model.attributes import ATTR_NAMES, GROUPS, INDEX, N_ATTR
from ..model.positions import FORMATIONS, GROUP, RELATED, SIDE, Pos
from .names import TEAM_NAMES, player_name
from .schema import GROUP_MODELS, Development, Foot, League, Player, Stadium, Tactics, Team

REFERENCE_DATE = date(2026, 9, 1)

# 位置原型：相對於能力水準 L 的偏移（點）。未列出的屬性偏移為 0。
_OFFSETS: Dict[str, Dict[str, float]] = {
    "GK": {"acceleration": -8, "sprint_speed": -12, "balance": -5, "strength": -2, "jumping": 3,
           "stamina": -10, "first_touch": -15, "dribbling": -35, "shielding": -35,
           "short_passing": -12, "long_passing": -15, "crossing": -40, "technique": -18,
           "finishing": -45, "long_shots": -40, "shot_power": -20, "heading": -35,
           "tackling": -40, "marking": -40, "def_positioning": -30, "vision": -15,
           "composure": 2, "concentration": 5, "off_the_ball": -35, "teamwork": -5,
           "work_rate": -20, "aggression": -15, "bravery": 5, "flair": -20,
           "set_piece_delivery": -40, "free_kick": -40, "penalty": -35, "long_throw": -30},
    "CB": {"heading": 10, "jumping": 7, "strength": 10, "marking": 10, "tackling": 10,
           "def_positioning": 10, "anticipation": 4, "concentration": 5, "bravery": 8,
           "aggression": 5, "decisions": 2, "teamwork": 3, "dribbling": -12, "finishing": -22,
           "long_shots": -15, "crossing": -14, "flair": -15, "technique": -8, "first_touch": -5,
           "vision": -8, "off_the_ball": -18, "acceleration": -5, "sprint_speed": -3,
           "agility": -10, "stamina": -3, "short_passing": -2, "shot_power": -5,
           "set_piece_delivery": -15, "free_kick": -18, "penalty": -12, "shielding": -2},
    "FB": {"stamina": 8, "acceleration": 6, "sprint_speed": 6, "agility": 2, "crossing": 5,
           "tackling": 3, "marking": 2, "def_positioning": 2, "work_rate": 6, "teamwork": 3,
           "heading": -6, "jumping": -4, "strength": -4, "finishing": -18, "long_shots": -10,
           "shot_power": -6, "off_the_ball": -6, "vision": -4, "flair": -6, "dribbling": -2,
           "technique": -4, "first_touch": -2, "short_passing": -1, "long_passing": -2,
           "set_piece_delivery": -6, "free_kick": -12, "penalty": -12, "shielding": -8},
    "DM": {"tackling": 7, "def_positioning": 8, "anticipation": 6, "teamwork": 6, "work_rate": 5,
           "marking": 4, "short_passing": 3, "long_passing": 3, "composure": 3, "decisions": 4,
           "concentration": 4, "strength": 3, "stamina": 4, "aggression": 3, "finishing": -14,
           "off_the_ball": -8, "dribbling": -4, "crossing": -8, "flair": -6, "sprint_speed": -4,
           "acceleration": -2, "agility": -2, "long_shots": -4, "first_touch": 1, "vision": 1,
           "shielding": 2, "set_piece_delivery": -6, "free_kick": -8, "penalty": -6},
    "CM": {"short_passing": 6, "long_passing": 4, "vision": 5, "decisions": 4, "stamina": 7,
           "work_rate": 5, "teamwork": 4, "first_touch": 4, "technique": 3, "composure": 2,
           "anticipation": 2, "tackling": -2, "marking": -5, "def_positioning": -2,
           "finishing": -8, "heading": -6, "jumping": -4, "strength": -3, "crossing": -2,
           "long_shots": 2, "sprint_speed": -2, "agility": 1, "balance": 1, "free_kick": -2,
           "penalty": -2, "shielding": -2},
    "AM": {"vision": 10, "technique": 8, "short_passing": 6, "dribbling": 8, "flair": 10,
           "first_touch": 7, "long_shots": 6, "finishing": 2, "off_the_ball": 5, "decisions": 3,
           "composure": 3, "agility": 5, "balance": 4, "acceleration": 2, "tackling": -16,
           "marking": -18, "def_positioning": -14, "strength": -8, "heading": -10, "jumping": -6,
           "work_rate": -4, "aggression": -6, "stamina": -2, "bravery": -6, "crossing": 2,
           "long_passing": 2, "shot_power": 2, "set_piece_delivery": 6, "free_kick": 6,
           "penalty": 4, "shielding": -2, "concentration": -2, "anticipation": 2},
    "W": {"acceleration": 11, "sprint_speed": 11, "agility": 9, "dribbling": 10, "crossing": 6,
          "flair": 7, "balance": 6, "first_touch": 3, "technique": 3, "off_the_ball": 4,
          "finishing": -2, "tackling": -14, "marking": -16, "def_positioning": -14,
          "heading": -12, "jumping": -6, "strength": -10, "aggression": -6, "bravery": -5,
          "long_passing": -3, "vision": 1, "decisions": -1, "composure": -1,
          "concentration": -3, "stamina": 2, "teamwork": -2, "shielding": -6,
          "set_piece_delivery": 2},
    "ST": {"finishing": 13, "off_the_ball": 11, "composure": 5, "heading": 5, "first_touch": 4,
           "shot_power": 5, "long_shots": 3, "anticipation": 5, "strength": 3, "jumping": 3,
           "acceleration": 4, "sprint_speed": 3, "dribbling": 2, "shielding": 6, "technique": 1,
           "tackling": -18, "marking": -20, "def_positioning": -18, "crossing": -8,
           "long_passing": -8, "short_passing": -3, "vision": -2, "work_rate": -2,
           "teamwork": -3, "stamina": -2, "concentration": -2, "flair": 2, "bravery": 2,
           "agility": 1, "balance": 2, "set_piece_delivery": -8, "free_kick": -2, "penalty": 5},
}
OFFSET_VEC: Dict[str, np.ndarray] = {}
for _g, _offs in _OFFSETS.items():
    _v = np.zeros(N_ATTR)
    for _k, _o in _offs.items():
        _v[INDEX[_k]] = _o
    OFFSET_VEC[_g] = _v

HEIGHT = {"GK": (190, 4.5), "CB": (187, 4.5), "FB": (178, 5), "DM": (181, 5), "CM": (180, 5),
          "AM": (176, 5), "W": (175, 5), "ST": (183, 6)}

_IDX = {g: np.array([INDEX[n] for n in names]) for g, names in GROUPS.items()}
_TECH = np.concatenate([_IDX["ball"], _IDX["shooting"], _IDX["set_pieces"]])
_PACE = np.array([INDEX["acceleration"], INDEX["sprint_speed"], INDEX["agility"]])
_ENDUR = np.array([INDEX["stamina"], INDEX["strength"], INDEX["jumping"]])
_MENTAL_EXP = np.array([INDEX[n] for n in ("anticipation", "decisions", "composure", "vision",
                                            "concentration", "teamwork", "def_positioning",
                                            "marking", "leadership")])

# 一支 25 人名單的位置組成
SQUAD = ([Pos.GK] * 3 + [Pos.CB] * 4 + [Pos.LB] * 2 + [Pos.RB] * 2 + [Pos.DM] * 2 + [Pos.CM] * 4
         + [Pos.AM] * 1 + [Pos.LW] * 2 + [Pos.RW] * 2 + [Pos.ST] * 3)


def _age_adjust(vec: np.ndarray, age: float, group: str, nat_fit: float) -> np.ndarray:
    """年齡效果（§8 的示意曲線）：身體先衰退、技術其次、心智隨經驗成長"""
    v = vec.copy()
    keep = 1.3 - nat_fit / 100  # 恢復力高 → 衰退慢
    if age > 27:
        a = age - 27
        v[_PACE] -= (1.2 * a + 0.25 * a * a) * keep
    if age > 29:
        a = age - 29
        v[_ENDUR] -= (0.8 * a + 0.15 * a * a) * keep
    if age > 31:
        a = age - 31
        v[_TECH] -= 0.5 * a + 0.1 * a * a
    if age < 22:
        v[_TECH] -= 1.5 * (22 - age)
    if age < 24:
        v[_MENTAL_EXP] -= 2.0 * (24 - age)
    elif age > 27:
        v[_MENTAL_EXP] += 0.5 * min(age - 27, 6)
    if group == "GK":
        gk = _IDX["goalkeeping"]
        if age < 25:
            v[gk] -= 2.0 * (25 - age)
        elif age > 33:
            a = age - 33
            v[gk] -= 0.5 * a + 0.3 * a * a
    return v


def _foot(rng: np.random.Generator, pos: Pos) -> Foot:
    side = SIDE.get(pos)
    p_left = {"L": 0.65 if GROUP[pos] == "FB" else 0.45, "R": 0.12}.get(side, 0.2)
    u = rng.random()
    if u < 0.03:
        return Foot.BOTH
    return Foot.LEFT if u < 0.03 + p_left else Foot.RIGHT


def _traits(rng: np.random.Generator, pos: Pos, a: Dict[str, int], foot: Foot) -> Dict[str, float]:
    g = GROUP[pos]
    t: Dict[str, float] = {}

    def maybe(name: str, prob: float) -> None:
        if rng.random() < prob:
            t[name] = round(float(rng.uniform(0.5, 1.0)), 2)

    if g in ("CM", "AM", "W", "ST") and a["long_shots"] >= 72:
        maybe("shoots_from_distance", 0.5)
    if g in ("CM", "AM", "DM") and a["vision"] >= 74:
        maybe("tries_killer_balls", 0.5)
    if a["dribbling"] >= 74 and a["flair"] >= 65:
        maybe("takes_on", 0.6)
    if g == "W" and ((SIDE.get(pos) == "L" and foot == Foot.RIGHT)
                     or (SIDE.get(pos) == "R" and foot == Foot.LEFT)):
        maybe("cuts_inside", 0.7)
    if g in ("FB", "W") and a["crossing"] >= 70:
        maybe("crosses_early", 0.3)
    if g == "ST" and a["off_the_ball"] >= 72 and a["acceleration"] >= 70:
        maybe("plays_on_shoulder", 0.5)
    if g in ("CM", "AM") and a["off_the_ball"] >= 65:
        maybe("arrives_late", 0.35)
    if g in ("CB", "FB", "DM") and a["aggression"] >= 70:
        maybe("dives_into_tackles", 0.35)
    if g == "GK" and a["gk_sweeping"] >= 70:
        maybe("rushes_out", 0.5)
    return t


def generate_player(rng: np.random.Generator, pos: Pos, level: float, pid: str,
                    age: Optional[float] = None) -> Player:
    g = GROUP[pos]
    if age is None:
        age = float(np.clip(rng.normal(26.5, 4.0), 17, 37))
    birth = REFERENCE_DATE - timedelta(days=int(age * 365.25))

    # 身材
    h_mu, h_sd = HEIGHT[g]
    height = int(np.clip(round(rng.normal(h_mu, h_sd)), 160, 204))
    bmi = float(np.clip(rng.normal(22.8, 0.9), 20.0, 26.0))
    weight = round(bmi * (height / 100) ** 2, 1)

    # 屬性 = 水準 + 原型偏移 + 群組相關雜訊 + 個別雜訊
    vec = np.full(N_ATTR, float(level)) + OFFSET_VEC[g]
    for gname in ("physical", "ball", "shooting", "defending", "mental", "set_pieces"):
        vec[_IDX[gname]] += rng.normal(0, 3.0)
    vec += rng.normal(0, 5.0, N_ATTR)

    # 身材效果：高個子較不敏捷、頭槌較好；體重 → 力量
    dh = height - h_mu
    vec[INDEX["agility"]] -= 0.35 * dh
    vec[INDEX["acceleration"]] -= 0.20 * dh
    vec[INDEX["heading"]] += 0.20 * dh
    vec[INDEX["strength"]] += 1.2 * (bmi - 22.8) + 0.15 * dh

    # 守門屬性：外野球員一律很低（絕對值）；門將 +3
    gk = _IDX["goalkeeping"]
    if g == "GK":
        vec[gk] = level + 3 + rng.normal(0, 3.0) + rng.normal(0, 4.0, len(gk))
    else:
        vec[gk] = rng.normal(15, 5, len(gk))

    # 定位球專家
    if g in ("CM", "AM", "W", "FB") and rng.random() < 0.15:
        vec[[INDEX["set_piece_delivery"], INDEX["free_kick"]]] += 12
    vec[INDEX["long_throw"]] = level - 20 + rng.normal(0, 10) + (30 if rng.random() < 0.05 else 0)

    # 隱藏屬性：與能力水準無關
    hid = {"consistency": (62, 14), "big_match": (58, 15), "injury_proneness": (40, 15),
           "temperament": (60, 15), "sportsmanship": (60, 15), "dirtiness": (40, 15),
           "professionalism": (60, 15), "ambition": (60, 15), "adaptability": (60, 15),
           "versatility": (55, 15)}
    for k, (mu, sd) in hid.items():
        vec[INDEX[k]] = rng.normal(mu, sd)
    if g in ("CB", "DM"):
        vec[INDEX["dirtiness"]] += 8
    vec[INDEX["leadership"]] = rng.normal(52, 14) + (8 if age > 29 else 0)

    nat_fit = float(np.clip(vec[INDEX["natural_fitness"]], 1, 99))
    vec = _age_adjust(vec, age, g, nat_fit)
    vals = np.clip(np.rint(vec), 1, 99).astype(int)
    a = {n: int(vals[i]) for i, n in enumerate(ATTR_NAMES)}

    groups = {gname: model(**{n: a[n] for n in GROUPS[gname]})
              for gname, model in GROUP_MODELS.items()}

    foot = _foot(rng, pos)
    weak = int(np.clip(rng.normal(85, 6) if foot == Foot.BOTH else rng.normal(45, 15), 1, 99))

    fam: Dict[Pos, int] = {p: int(rng.integers(15, 36)) for p in Pos}
    if g != "GK":
        fam[Pos.GK] = int(rng.integers(1, 10))
    else:
        fam = {p: int(rng.integers(1, 10)) for p in Pos}
    for p_rel, v_rel in RELATED[pos].items():
        fam[p_rel] = int(np.clip(v_rel + rng.normal(0, 8), 20, 95))
    fam[pos] = 100

    growth = max(0.0, (24 - age)) * float(rng.uniform(1.0, 3.0))
    potential = int(np.clip(round(level + growth + rng.normal(0, 2)), 1, 99))

    return Player(
        player_id=pid, name=player_name(rng), birth_date=birth, nationalities=["XX"],
        position=pos, height_cm=height, weight_kg=weight, preferred_foot=foot, weak_foot=weak,
        positions=fam, traits=_traits(rng, pos, a, foot),
        development=Development(potential=max(potential, 1)), **groups,
    )


def _tactics_for(rng: np.random.Generator, rel_level: float) -> Tactics:
    """強隊傾向高位、壓迫、短傳組織；弱隊傾向低位、直接"""
    def lvl(base: float) -> int:
        return int(np.clip(round(base + rng.normal(0, 0.7)), 1, 5))

    formation = str(rng.choice(list(FORMATIONS), p=[0.35, 0.35, 0.15, 0.15]))
    strong = float(np.clip(rel_level / 4.0, -1.5, 1.5))
    return Tactics(
        formation=formation,
        mentality=lvl(3 + 0.6 * strong),
        line_height=lvl(3 + 0.7 * strong),
        press=lvl(3 + 0.7 * strong),
        width=lvl(3),
        tempo=lvl(3),
        directness=lvl(3 - 0.7 * strong),
        shooting=lvl(3),
        buildup="short" if strong > 0.4 else ("long" if strong < -0.6 else "mixed"),
        tackling=str(rng.choice(["cautious", "normal", "aggressive"], p=[0.2, 0.6, 0.2])),
        offside_trap=bool(rng.random() < 0.25 + 0.15 * strong),
    )


def generate_team(rng: np.random.Generator, team_id: str, name: str, level: float,
                  league_mean: float = 65.0) -> Team:
    players: List[Player] = []
    for i, pos in enumerate(SQUAD):
        # 前段（主力）約為球隊水準，後段（輪替）略低
        starter_like = i in (0, 3, 4, 7, 9, 11, 13, 15, 16, 19, 20, 21, 22, 23)
        lv = level + rng.normal(0, 2.5) - (0 if starter_like else 3.0)
        players.append(generate_player(rng, pos, lv, f"{team_id}-{i + 1:02d}"))
    tactics = _tactics_for(rng, level - league_mean)
    return Team(team_id=team_id, name=name, short_name=name[:2], level=round(level, 2),
                players=players, tactics=tactics,
                stadium=Stadium(name=f"{name}體育場",
                                capacity=int(rng.integers(15, 75)) * 1000,
                                attendance_ratio=float(np.clip(rng.normal(0.9, 0.06), 0.6, 1.0))))


def generate_league(seed: int = 1, n_teams: int = 20, mean: float = 65.0, spread: float = 3.5,
                    name: str = "模擬聯賽", season: str = "2026/27") -> League:
    """spread：球隊水準的標準差（點）。3.5 是 M4 校準值（熱門勝率與半場領先勝率的折衷，§6.4 L4）。"""
    rng = np.random.default_rng(seed)
    levels = np.sort(rng.normal(mean, spread, n_teams))[::-1]
    names = list(TEAM_NAMES[:n_teams]) if n_teams <= len(TEAM_NAMES) else \
        [f"球隊{i + 1}" for i in range(n_teams)]
    teams = [generate_team(rng, f"T{i + 1:02d}", names[i], float(levels[i]), mean)
             for i in range(n_teams)]
    return League(name=name, season=season, teams=teams)
