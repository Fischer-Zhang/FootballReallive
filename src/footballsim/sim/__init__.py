"""v2：逐幀（10 Hz）足球比賽引擎。與 v1（footballsim.engine，事件驅動）平行；v1 作為校準基準保留。

模組：world（狀態）、physics（球與移動）、phase（攻防階段）、shape（目標點）、play（接觸、決策、動作）、
rules（出界、進球、重啟）、engine（主迴圈與結果）。屬性、參數、行為模型與統計沿用 v1。
"""
from .engine import TickMatch, simulate_tick_match  # noqa: F401
