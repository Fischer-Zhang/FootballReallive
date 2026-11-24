"""虛構名稱產生器：避免使用真實球員與球隊（設計文件 §3.13）"""
from __future__ import annotations

import numpy as np

_SYL = ("ka", "lo", "ren", "mi", "sa", "to", "vel", "an", "dor", "ri", "ne", "bas", "tel", "or",
        "fin", "ga", "lu", "mar", "zen", "ko", "pe", "ra", "sol", "vin", "da", "el", "is", "mu",
        "tor", "ya", "ber", "cas", "dio", "fe", "gor", "hal", "jo", "kel", "len", "mo", "nor", "ol")

TEAM_NAMES = ("青山聯", "海港城", "北辰競技", "南嶺流浪者", "東岬", "西川體育", "鐵橋", "白楊",
              "赤崖", "金沙", "銀湖", "雲杉", "霧峰", "星原", "風城", "雷岩", "松濤", "楓谷",
              "石岸", "藍灣", "曙光", "黑森", "長堤", "翠嶼")


def _word(rng: np.random.Generator, lo: int, hi: int) -> str:
    n = int(rng.integers(lo, hi + 1))
    return "".join(str(rng.choice(_SYL)) for _ in range(n)).capitalize()


def player_name(rng: np.random.Generator) -> str:
    return f"{_word(rng, 1, 2)} {_word(rng, 2, 3)}"
