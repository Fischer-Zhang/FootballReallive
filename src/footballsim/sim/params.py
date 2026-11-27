"""v2 參數（config/sim_params.yaml）"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

import yaml

from ..config import CONFIG_DIR, _to_ns, deep_update


def load_sim_params(path: str | Path | None = None, overrides: Mapping | None = None):
    with open(path or CONFIG_DIR / "sim_params.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if overrides:
        raw = deep_update(copy.deepcopy(raw), overrides)
    sp = _to_ns(raw)
    sp._raw = raw
    return sp
