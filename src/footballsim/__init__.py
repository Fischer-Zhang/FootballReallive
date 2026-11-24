"""footballsim — 屬性驅動、以真實資料校準的足球比賽模擬引擎"""
import os as _os

# 低負荷：數值函式庫一律單執行緒（否則每個行程可能各開十幾條執行緒）。必須在載入 numpy 之前設定。
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")

__version__ = "0.1.0"
