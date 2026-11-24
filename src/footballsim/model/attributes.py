"""60 項能力屬性的唯一定義來源。

schema（資料驗證）、generator（球員生成）、engine（向量運算）都由這裡取得屬性順序與索引，
因此新增或調整屬性只需要改這一個檔案（schema 會由測試檢查是否一致）。
"""
from __future__ import annotations

from enum import IntEnum
from types import SimpleNamespace
from typing import Dict, Mapping, Tuple

import numpy as np

GROUPS: Dict[str, Tuple[str, ...]] = {
    "physical": ("acceleration", "sprint_speed", "agility", "balance", "strength",
                 "jumping", "stamina", "natural_fitness"),
    "ball": ("first_touch", "dribbling", "shielding", "short_passing", "long_passing",
             "crossing", "technique"),
    "shooting": ("finishing", "long_shots", "shot_power", "heading"),
    "defending": ("tackling", "marking", "def_positioning"),
    "mental": ("anticipation", "decisions", "vision", "composure", "concentration",
               "off_the_ball", "teamwork", "work_rate", "aggression", "bravery", "flair",
               "leadership", "determination"),
    "set_pieces": ("set_piece_delivery", "free_kick", "penalty", "long_throw"),
    "goalkeeping": ("gk_reflexes", "gk_diving", "gk_positioning", "gk_handling", "gk_one_on_one",
                    "gk_aerial", "gk_sweeping", "gk_kicking", "gk_throwing", "gk_communication",
                    "gk_penalty"),
    "hidden": ("consistency", "big_match", "injury_proneness", "temperament", "sportsmanship",
               "dirtiness", "professionalism", "ambition", "adaptability", "versatility"),
}

ATTR_NAMES: Tuple[str, ...] = tuple(n for g in GROUPS.values() for n in g)
N_ATTR = len(ATTR_NAMES)
ATTR = IntEnum("ATTR", [n.upper() for n in ATTR_NAMES], start=0)
INDEX: Dict[str, int] = {n: i for i, n in enumerate(ATTR_NAMES)}
GROUP_OF: Dict[str, str] = {n: g for g, names in GROUPS.items() for n in names}

LABELS_ZH: Dict[str, str] = {
    "acceleration": "加速", "sprint_speed": "速度", "agility": "敏捷", "balance": "平衡",
    "strength": "力量", "jumping": "彈跳", "stamina": "耐力", "natural_fitness": "恢復力",
    "first_touch": "停球", "dribbling": "盤球", "shielding": "護球", "short_passing": "短傳",
    "long_passing": "長傳", "crossing": "傳中", "technique": "腳法",
    "finishing": "終結", "long_shots": "遠射", "shot_power": "射門力量", "heading": "頭槌",
    "tackling": "搶斷", "marking": "盯人", "def_positioning": "防守站位",
    "anticipation": "預判", "decisions": "決策", "vision": "視野", "composure": "冷靜",
    "concentration": "專注", "off_the_ball": "無球跑位", "teamwork": "團隊配合",
    "work_rate": "投入度", "aggression": "侵略性", "bravery": "勇敢", "flair": "創造力",
    "leadership": "領導力", "determination": "鬥志",
    "set_piece_delivery": "定位球傳送", "free_kick": "直接自由球", "penalty": "十二碼",
    "long_throw": "長界外球",
    "gk_reflexes": "反應", "gk_diving": "撲救範圍", "gk_positioning": "門將站位",
    "gk_handling": "接球穩定", "gk_one_on_one": "單刀", "gk_aerial": "制空",
    "gk_sweeping": "清道夫", "gk_kicking": "大腳", "gk_throwing": "手拋球",
    "gk_communication": "指揮", "gk_penalty": "撲十二碼",
    "consistency": "穩定性", "big_match": "大場面", "injury_proneness": "易傷性",
    "temperament": "脾氣", "sportsmanship": "運動精神", "dirtiness": "骯髒度",
    "professionalism": "職業態度", "ambition": "企圖心", "adaptability": "適應力",
    "versatility": "多面性",
}


def _idx(*names: str) -> np.ndarray:
    return np.array([INDEX[n] for n in names], dtype=np.intp)


# ---- 疲勞作用分組（§4.13）：身體全額、技術半額、心智依心智曲線 ----
PHYSICAL_FATIGUE = _idx("acceleration", "sprint_speed", "agility", "balance", "strength", "jumping")
TECHNICAL_FATIGUE = _idx("first_touch", "dribbling", "shielding", "short_passing", "long_passing",
                         "crossing", "technique", "finishing", "long_shots", "shot_power", "heading",
                         "tackling", "set_piece_delivery", "free_kick", "penalty", "long_throw")
MENTAL_FATIGUE = _idx("anticipation", "decisions", "vision", "composure", "concentration",
                      "off_the_ball", "teamwork", "marking", "def_positioning")
MENTAL_GROUP = _idx(*GROUPS["mental"])
# 當日狀態（穩定性）作用在所有非隱藏屬性上
DAY_FORM = _idx(*[n for n in ATTR_NAMES if GROUP_OF[n] != "hidden"])


def compile_composites(spec: Mapping[str, Mapping[str, float]]) -> tuple[np.ndarray, SimpleNamespace]:
    """把 {複合能力: {屬性: 權重}} 編譯成 (60 × K) 權重矩陣；複合能力 = eff @ W。

    回傳的索引物件可以用 C.SHORT_PASS 這種寫法取得欄位位置。權重自動正規化為 1。
    """
    names = list(spec)
    w = np.zeros((N_ATTR, len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        weights = spec[name]
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError(f"複合能力 {name} 的權重總和必須 > 0")
        for attr, val in weights.items():
            if attr not in INDEX:
                raise KeyError(f"複合能力 {name} 使用了未知屬性 {attr}")
            w[INDEX[attr], j] = val / total
    return w, SimpleNamespace(**{n.upper(): j for j, n in enumerate(names)})
