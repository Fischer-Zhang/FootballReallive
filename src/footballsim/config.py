"""設定檔載入：所有引擎係數都放在 config/engine_params.yaml（設計原則 P6）"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("FOOTBALLSIM_CONFIG_DIR", PROJECT_ROOT / "config"))
DEFAULT_PARAMS = CONFIG_DIR / "engine_params.yaml"


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        if all(isinstance(k, str) and k.isidentifier() for k in obj):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        return {k: _to_ns(v) for k, v in obj.items()}   # 例如 {60: 1} 這種數字鍵
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def deep_update(base: dict, updates: Mapping) -> dict:
    """以點號路徑或巢狀 dict 覆寫參數，例如 {"shot.k_gk": 0.25}"""
    for key, val in updates.items():
        if isinstance(key, str) and "." in key:
            node = base
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = val
        elif isinstance(val, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], val)
        else:
            base[key] = val
    return base


FITTED = CONFIG_DIR / "fitted.yaml"          # StatsBomb 2015/16 事件資料擬合
FITTED360 = CONFIG_DIR / "fitted360.yaml"    # StatsBomb 360 位置資料擬合（行為模型）


def load_raw(path: str | Path | None = None, fitted: bool = True) -> dict:
    """載入引擎參數；資料擬合結果（fitted.yaml、fitted360.yaml）依序覆蓋同名項目"""
    with open(path or DEFAULT_PARAMS, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if fitted and path is None:
        for f in (FITTED, FITTED360):
            if f.exists():
                with open(f, encoding="utf-8") as fh:
                    fit = yaml.safe_load(fh) or {}
                for k in [k for k in fit if k.startswith("_")]:
                    fit.pop(k)
                raw = deep_update(raw, fit)
    return raw


def load_params(path: str | Path | None = None, overrides: Mapping | None = None) -> SimpleNamespace:
    """載入引擎參數。回傳可用屬性存取的物件，原始 dict 放在 `_raw`。"""
    raw = load_raw(path)
    if overrides:
        raw = deep_update(copy.deepcopy(raw), overrides)
    params = _to_ns(raw)
    params._raw = raw
    return params
