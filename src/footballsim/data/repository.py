"""聯賽資料的 JSON 讀寫"""
from __future__ import annotations

from pathlib import Path

from .schema import League


def save_league(league: League, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(league.model_dump_json(indent=1), encoding="utf-8")
    return path


def load_league(path: str | Path) -> League:
    return League.model_validate_json(Path(path).read_text(encoding="utf-8"))
