"""以 StatsBomb 360 資料擬合站位相關的模型（M4 站位校準）

產出：
- config/box_runs.yaml：球在前場時，禁區與禁區前沿的「真實攻方站位組合」。
  引擎在同樣的持球位置下抽一組真實組合，讓跑位者去佔那些點（shape.compute 的禁區跑位）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from ..config import CONFIG_DIR
from . import statsbomb360 as s3

SOURCE = "StatsBomb Open Data 360（UEFA Euro 2024、FIFA World Cup 2022，各隨機 25 場）"
CY = 34.0
LENGTH, WIDTH = 105.0, 68.0
BOX_GRID = np.column_stack([g.ravel() for g in np.meshgrid(np.linspace(89, 104.5, 6), np.linspace(14.5, 53.5, 8))])
ZONE_X = 80.0          # 禁區與禁區前沿：x ≥ 80 m
ZONE_HALF = 22.0       # |y − 34| ≤ 22 m


def box_visible(poly: np.ndarray, min_share: float = 0.9) -> bool:
    return bool(s3.inside(poly, BOX_GRID).mean() >= min_share)


def box_run_pool(frames, min_x: float = 70.0) -> list:
    """球在前場（x ≥ min_x）時，禁區可完整看到的畫面中，攻方（不含持球者、門將）在禁區＋前沿的位置。
    y 以「球所在的一側為正」標準化。回傳 [[球 x, 球 |y−34|, [[x, y_rel], ...]], ...]"""
    df = frames[frames.kind.isin(["pass", "cross", "carry"]) & (frames.x >= min_x)]
    out = []
    for r in df.itertuples():
        if not box_visible(s3.parse_poly(r.vis)):
            continue
        f = s3.parse_frame(r.frame)
        T = f["T"]
        side = 1.0 if r.y >= CY else -1.0
        m = (T[:, 0] >= ZONE_X) & (np.abs(T[:, 1] - CY) <= ZONE_HALF)
        spots = [[round(float(x), 1), round(float((y - CY) * side), 1)] for x, y in T[m]]
        out.append([round(float(r.x), 1), round(abs(float(r.y) - CY), 1), spots])
    return out


def save_box_pool(pool: list, path: Path = CONFIG_DIR / "box_runs.yaml") -> Path:
    head = {"_source": SOURCE,
            "_note": "球在前場時禁區＋前沿（x ≥ 80 m、|y−34| ≤ 22 m）的攻方站位；每列 [球 x, 球 |y−34|, [[x, y_rel], ...]]，"
                     "y_rel 以球所在側為正"}
    lines = ["# 由 `python -m footballsim statsbomb fit360` 產生，請勿手動修改",
             yaml.safe_dump(head, allow_unicode=True, sort_keys=False).rstrip(), "frames:"]
    lines += [f"- {json.dumps(f, separators=(',', ':'))}" for f in pool]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# 攻方橫向站位：球的橫向位置 → 球前方（8 m 以上）與同線（±8 m）的隊友在橫向的分布
LAT_ZONES = [35.0, 70.0]                    # 球 x 分區：本方、中場、前場
LAT_BALL = [4.0, 9.0, 15.0, 22.0]           # 球離中線的距離分組（5 組）
# 各組：(球員來源, 相對球的縱向範圍 m)。T = 攻方的隊友，O = 防守方外野球員（縱向以攻方座標、往對方球門為正）
LAT_GROUPS = {"ahead": ("T", 8.0, 24.0), "level": ("T", -8.0, 8.0),
              "def_near": ("O", 8.0, 24.0), "def_far": ("O", 24.0, 45.0), "def_level": ("O", -8.0, 8.0)}
LAT_EDGES = np.arange(-34.0, 34.01, 2.0)    # 有號橫向位置（球在 + 側），每 2 m 一格


def fit_att_lateral(frames, prior_area: float = 400.0) -> dict:
    """攻守雙方的橫向站位（360）：球在傳球／傳中時，球前方 8–24 m（ahead）與同線 ±8 m（level）的隊友，
    以及防守方外野球員（球與對方球門之間 8–24 m、24–45 m，與同線 ±8 m）在橫向的分布，
    依球的 x 分區與離中線的距離分組，橫向座標以球所在側為正。
    可視範圍不涵蓋全場，所以不能直接數人：每格的密度 = 看到的人數 ÷ 看得到的面積（可視多邊形 ∩ 球場 ∩ 該縱向範圍），
    稀疏的格子以 prior_area（m²）的偽面積向該組平均密度收縮。分布 = 密度（每格面積相同）正規化。
    引擎依陣型的左右順序把球員放在這個分布的分位數上（shape._att_lateral）。"""
    nz, nb, ns = len(LAT_ZONES) + 1, len(LAT_BALL) + 1, len(LAT_EDGES) - 1
    H = {g: np.zeros((nz, nb, ns)) for g in LAT_GROUPS}
    A = {g: np.zeros((nz, nb, ns)) for g in LAT_GROUPS}
    N = np.zeros((nz, nb))
    mid = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2
    for r in frames[frames.kind.isin(["pass", "cross"])].itertuples():
        poly = s3.parse_poly(r.vis)
        if len(poly) < 3:
            continue
        bx, by = float(r.x), float(r.y)
        z = int(np.digitize(bx, LAT_ZONES))
        b = int(np.digitize(abs(by - CY), LAT_BALL))
        sg = 1.0 if by >= CY else -1.0
        fr_ = s3.parse_frame(r.frame)
        N[z, b] += 1
        for g, (who, lo, hi) in LAT_GROUPS.items():
            T = fr_[who]
            gd = np.arange(lo + 1.0, hi, 2.0)                      # 縱向取樣（每格 2 m × 2 m = 4 m²）
            GD, GS = np.meshgrid(gd, mid, indexing="ij")
            px, py = bx + GD, CY + GS * sg
            inb = (px >= 0) & (px <= LENGTH) & (py >= 0) & (py <= WIDTH)
            vis = np.zeros(GD.shape)
            if inb.any():
                vis[inb] = s3.inside(poly, np.column_stack([px[inb], py[inb]])) * 4.0
            A[g][z, b] += vis.sum(0)
            if len(T):
                m = (T[:, 0] - bx >= lo) & (T[:, 0] - bx < hi)
                si = np.clip(np.digitize((T[m, 1] - CY) * sg, LAT_EDGES[1:-1]), 0, ns - 1)
                np.add.at(H[g][z, b], si, 1.0)
    out = {"zones": LAT_ZONES, "ball": LAT_BALL, "edges": [float(v) for v in LAT_EDGES], "_frames": N.astype(int).tolist()}
    for g in LAT_GROUPS:
        prior = H[g].sum((0, 1, 2)) / max(A[g].sum(), 1.0)          # 該組的平均密度
        dens = (H[g] + prior_area * prior) / (A[g] + prior_area)
        # 沿橫向做小幅平滑（相鄰三格 1-2-1），去掉格與格之間的雜訊
        pad = np.pad(dens, ((0, 0), (0, 0), (1, 1)), mode="edge")
        sm = (pad[..., :-2] + 2 * pad[..., 1:-1] + pad[..., 2:]) / 4
        pdf = sm / sm.sum(-1, keepdims=True)
        out[g] = [[[round(float(v), 5) for v in pdf[i, j]] for j in range(nb)] for i in range(nz)]
    return out


def save_att_lateral(doc: dict, path: Path = CONFIG_DIR / "att_lateral.yaml") -> Path:
    head = {"_source": SOURCE, "_note": "橫向站位：[球 x 分區][球離中線距離分組][橫向每 2 m 一格 −34~34，球所在側為正] 的機率；"
                                        "ahead = 球前方 8–24 m 的隊友，level = 同線 ±8 m 的隊友；def_near／def_far／def_level = "
                                        "防守方外野球員在球與對方球門之間 8–24 m、24–45 m 與同線 ±8 m（fit360 產生）"}
    path.write_text("# 由 `python -m footballsim statsbomb fit360` 產生，請勿手動修改\n"
                    + yaml.safe_dump({**head, **doc}, allow_unicode=True, sort_keys=False, default_flow_style=None),
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------ 行為模型（以位置特徵擬合）
# StatsBomb 的 under_pressure 是人工標記「有人在逼搶」，與距離不等價：360 資料中最近對手在 4.6 m 內的
# 傳球只有 16–26% 被標記。引擎沒有這個標記，所以射門／傳中／解圍傾向改用引擎也算得出的位置特徵擬合：
# 最近對手距離（含門將）、射門路線上的防守人數與最近外野防守者距離（pitch.shot_context）。
NEAR_CAP = 8.0
GOAL_PTS = np.array([[104.0, 30.5], [104.0, 37.5], [97.0, 34.0]])


def geo_actions(frames):
    """持球動作（傳球、傳中、過人、射門、解圍）的位置特徵，定義與引擎相同"""
    import pandas as pd

    from ..engine.pitch import shot_context

    df = frames[frames.kind.isin(["pass", "cross", "take_on", "shot", "clear"])]
    rows = []
    for r in df.itertuples():
        f = s3.parse_frame(r.frame)
        O, K = f["O"], f["K"]
        opp = np.vstack([O, K]) if len(K) else O
        if len(opp) == 0:
            continue
        dn = float(np.hypot(opp[:, 0] - r.x, opp[:, 1] - r.y).min())
        n_cone, d_out, _ = shot_context(r.x, r.y, O)
        goal_vis = bool(s3.inside(s3.parse_poly(r.vis), GOAL_PTS).all())
        rows.append((r.kind, r.x, r.y, min(dn, NEAR_CAP), n_cone, d_out, goal_vis, r.body))
    return pd.DataFrame(rows, columns=["kind", "x", "y", "d_opp", "n_cone", "d_near", "goal_vis", "body"])


SHOT_DIST_BINS = [0.0, 6.0, 11.0, 16.5, 25.0, 32.0, 40.0]


def fit_shot_propensity(geo) -> dict:
    """進攻區（距球門 ≤ 40 m）持球時選擇射門的機率：
    logit = a0 + a_dist·距離 + a_logang·ln(夾角) + a_near·最近外野防守者距離。
    不納入「射門路線上的防守人數」：射門瞬間防守者已撲進射門路線封堵，這個特徵在射門事件上是結果而非原因
    （資料中路線上人越多、射門率反而越高）。對封堵人數的反應改由引擎的射門品質項處理（模型假設 k_quality）。
    只有球門區在畫面內的樣本能用（要看得到防守者），但那些樣本偏向容易射門的情境（距離 25–40 m 只有 37% 的決策看得到
    球門，其中射門率 2.4%，全部決策的射門率是 1.6%）。以「距離分段 × 是否射門」的保留比例的倒數加權，
    讓擬合的機率回到全部決策的射門率。"""
    from ..engine.core_math import shot_geometry
    from .fit import logistic_fit

    ev = geo[geo.kind.isin(["pass", "cross", "take_on", "shot"])]
    gx, gy = np.maximum(105 - ev.x.to_numpy(float), 0.3), np.abs(ev.y.to_numpy(float) - 34)
    geo_ = [shot_geometry(a, b) for a, b in zip(gx, gy)]
    dist = np.array([g[0] for g in geo_])
    keep = dist <= 40
    ev, dist = ev[keep], dist[keep]
    lang = np.log(np.maximum([g[1] for g in np.array(geo_, dtype=object)[keep]], 1e-4))
    shot = (ev.kind == "shot").to_numpy(float)
    vis = ev.goal_vis.to_numpy(bool)
    stratum = np.digitize(dist, SHOT_DIST_BINS[1:-1]) * 2 + shot.astype(int)
    n_all = np.bincount(stratum, minlength=2 * (len(SHOT_DIST_BINS) - 1)).astype(float)
    n_vis = np.bincount(stratum[vis], minlength=len(n_all)).astype(float)
    sw = (n_all / np.maximum(n_vis, 1.0))[stratum][vis]
    X = np.column_stack([np.ones(len(ev)), dist, lang, ev.d_near.to_numpy(float)])[vis]
    y = shot[vis]
    w = logistic_fit(X, y, sw=sw)
    return {"a0": round(float(w[0]), 4), "a_dist": round(float(w[1]), 4), "a_logang": round(float(w[2]), 4),
            "a_near": round(float(w[3]), 4), "_n": int(vis.sum()), "_shot_rate%": round(100 * float((y * sw).sum() / sw.sum()), 2),
            "_shot_rate_visible%": round(100 * float(y.mean()), 2), "_shot_rate_all%": round(100 * float(shot.mean()), 2)}


def fit_cross_propensity(geo) -> dict:
    """邊路進攻三區（x ≥ 80 m、離中線 ≥ 12 m）持球時選擇傳中的機率：
    logit = c0 + c_depth·距底線 + c_wide·離中路 + c_near·最近對手距離"""
    from .fit import logistic_fit

    ev = geo[geo.kind.isin(["pass", "cross", "take_on", "shot"])]
    ev = ev[(ev.x >= 80) & (np.abs(ev.y - 34) >= 12)]
    X = np.column_stack([np.ones(len(ev)), 105 - ev.x.to_numpy(float), np.abs(ev.y.to_numpy(float) - 34),
                         ev.d_opp.to_numpy(float)])
    y = (ev.kind == "cross").to_numpy(float)
    w = logistic_fit(X, y)
    return {"c0": round(float(w[0]), 4), "c_depth": round(float(w[1]), 4), "c_wide": round(float(w[2]), 4),
            "c_near": round(float(w[3]), 4), "_n": int(len(ev)), "_cross_rate%": round(100 * float(y.mean()), 2)}


def fit_clear_propensity(geo) -> dict:
    """本方三區（x ≤ 35 m）持球時選擇（腳）解圍的機率：logit = k0 + k_x·x + k_wide·|y−34| + k_near·最近對手距離。
    頭槌解圍是對空中球的反應，不屬於持球決策，不列入。"""
    from .fit import logistic_fit

    ev = geo[(geo.kind.isin(["pass", "take_on"]) | ((geo.kind == "clear") & (geo.body != "Head")))
             & (geo.x <= 35)]
    X = np.column_stack([np.ones(len(ev)), ev.x.to_numpy(float), np.abs(ev.y.to_numpy(float) - 34),
                         ev.d_opp.to_numpy(float)])
    y = (ev.kind == "clear").to_numpy(float)
    w = logistic_fit(X, y)
    return {"k0": round(float(w[0]), 4), "k_x": round(float(w[1]), 4), "k_wide": round(float(w[2]), 4),
            "k_near": round(float(w[3]), 4), "_n": int(len(ev)), "_clear_rate%": round(100 * float(y.mean()), 2)}


def fit_carry_propensity(frames) -> dict:
    """持球時選擇帶球（≥ 3 m）的機率：logit = g0 + g_x·x/10 + g_wide·|y−34|/10 + g_near·最近對手距離 + g_space·正前方空間。
    決策狀態 = 帶球 ≥ 3 m 的起點（帶球）與傳球、傳中、射門、過人、解圍（不帶球）；正前方 12 m 須在畫面內。
    StatsBomb 每次接球後的移動都記為帶球，短於 3 m 的屬於接球調整，引擎不視為決策，所以不列入。"""
    from ..engine.decision import space_ahead
    from .fit import logistic_fit

    ln = np.hypot(frames.ex - frames.x, frames.ey - frames.y)
    d = frames[((frames.kind == "carry") & (ln >= 3.0)) | frames.kind.isin(["pass", "cross", "shot", "take_on", "clear"])]
    rows = []
    for r in d.itertuples():
        f = s3.parse_frame(r.frame)
        O = np.vstack([f["O"], f["K"]]) if len(f["K"]) else f["O"]
        if len(O) == 0 or not s3.inside(s3.parse_poly(r.vis), np.array([[min(r.x + 12.0, 104.5), r.y]]))[0]:
            continue
        dn = float(np.hypot(O[:, 0] - r.x, O[:, 1] - r.y).min())
        rows.append((float(r.kind == "carry"), r.x / 10, abs(r.y - CY) / 10, min(dn, NEAR_CAP), space_ahead(r.x, r.y, O)))
    A = np.array(rows, float)
    X = np.column_stack([np.ones(len(A)), A[:, 1:]])
    w = logistic_fit(X, A[:, 0])
    return {"g0": round(float(w[0]), 4), "g_x": round(float(w[1]), 4), "g_wide": round(float(w[2]), 4),
            "g_near": round(float(w[3]), 4), "g_space": round(float(w[4]), 4), "_n": int(len(A)),
            "_carry_rate%": round(100 * float(A[:, 0].mean()), 2)}


# ------------------------------------------------------------------ 帶球方向選擇模型（條件 logit）
CARRY_CHOICE_MIN = 3.0       # 只取 ≥ 3 m 的帶球（與帶球傾向、引擎的決策定義相同）
CARRY_MATCH_DEG = 20.0       # 被選者 = 方向最接近真實帶球方向的選項，差距須在此角度內


def _standin_sim(params):
    """引擎的 MatchSim（平均能力：全員同速度、同預判），用來在真實站位上跑引擎自己的帶球選項生成"""
    from ..data.generator import generate_league
    from ..engine.match import MatchSim
    from ..model.attributes import ATTR

    lg = generate_league(seed=3, n_teams=4)
    sim = MatchSim(lg.teams[0], lg.teams[1], params=params, seed=1)
    sim.vmax[:] = float(sim.vmax.mean())
    sim.tau[:] = float(sim.tau.mean())
    sim.eff[:, ATTR.ANTICIPATION] = sim.R
    return sim


def carry_choice_data(frames, params=None):
    """每次 ≥ 3 m 的帶球建立選項集合：把畫面內的防守者放到引擎的 22 人陣列，呼叫引擎的 decision._carries
    （與引擎完全相同的方向、可帶距離與特徵）。只保留起點到終點都在可視範圍內的選項；
    被選者 = 方向最接近真實帶球的選項（必須在可視範圍內，且引擎認為帶得動）。
    回傳 (特徵 X, 組別 g, 是否被選 y, 統計)。"""
    from ..config import load_params
    from ..engine import decision

    p = params or load_params()
    sim = _standin_sim(p)
    c = sim.active_slots(0)[6]
    gk = sim.gk_slot(1)
    outs = [s for s in sim.active_slots(1) if s != gk]
    Xs, gs, ys = [], [], []
    g, n_state, n_skip = 0, 0, 0
    metas = []
    d = frames[frames.kind == "carry"]
    for r in d.itertuples():
        if not np.isfinite([r.ex, r.ey]).all() or np.hypot(r.ex - r.x, r.ey - r.y) < CARRY_CHOICE_MIN:
            continue
        f = s3.parse_frame(r.frame)
        O, K = f["O"], f["K"]
        if len(O) == 0:
            continue
        n_state += 1
        pos = np.full((22, 2), -100.0)
        pos[c] = (r.x, r.y)
        if len(K):
            pos[gk] = K[0]
        order = np.argsort(np.hypot(O[:, 0] - r.x, O[:, 1] - r.y))[: len(outs)]
        pos[outs[: len(order)]] = O[order]
        x = decision.make_context(sim, 0, c, pos, 90.0)
        x.carry_len = float(np.hypot(r.ex - r.x, r.ey - r.y))      # 真實想帶多遠（選項的距離 = 它與防守者追截限制的較小者）
        opts = []
        decision._carries(sim, x, opts)
        if len(opts) < 2:
            n_skip += 1
            continue
        poly = s3.parse_poly(r.vis)
        mid = np.array([[(r.x + o.x) / 2, (r.y + o.y) / 2] for o in opts])
        end = np.array([[o.x, o.y] for o in opts])
        vis = s3.inside(poly, mid) & s3.inside(poly, end)
        ang = np.degrees(np.arctan2(end[:, 1] - r.y, end[:, 0] - r.x))
        real = np.degrees(np.arctan2(r.ey - r.y, r.ex - r.x))
        diff = np.abs((ang - real + 180.0) % 360.0 - 180.0)
        diff[~vis] = np.inf
        i = int(diff.argmin())
        if diff[i] > CARRY_MATCH_DEG or vis.sum() < 2:
            n_skip += 1
            continue
        keep = np.flatnonzero(vis)
        Xs.append(np.vstack([opts[k].feat for k in keep]))
        gs.append(np.full(len(keep), g))
        y = np.zeros(len(keep))
        y[int(np.flatnonzero(keep == i)[0])] = 1.0
        ys.append(y)
        metas.append((r.x, r.y))
        g += 1
    return np.vstack(Xs), np.concatenate(gs), np.concatenate(ys), {"states": n_state, "skipped": n_skip, "meta": metas}


def fit_through_propensity(frames) -> dict:
    """身後球（把球送到對方最深防守者之後給防線附近的跑動隊友）的傾向：
    logit = g0 + g_x·x/10 + g_wide·|y−34|/10 + g_near·最近對手距離 + g_run·接球者數 + g_gap·（防線 − x）/10。
    狀態 = 有接球者的不帶球決策（傳球、傳中、過人、射門、解圍；與射門、傳中的樣本定義相同，帶球另有閘門）；
    接球者 = 防線（最深可見外野防守者，至少 52.5 m）前後 8 m 內、在持球者前方 2 m 以上的可見隊友（最多算 3 名）；
    被選 = 這是一次傳球（非傳中），落點在防線之後至少 1 m。與引擎的選項生成（decision.through_runners）定義相同。"""
    d = frames[frames.kind.isin(["pass", "cross", "take_on", "shot", "clear"])]
    rows = []
    for r in d.itertuples():
        f = s3.parse_frame(r.frame)
        O, K, T = f["O"], f["K"], f["T"]
        if len(O) < 3:
            continue
        opp = np.vstack([O, K]) if len(K) else O
        line = max(float(O[:, 0].max()), 52.5)
        if r.x >= line or 105.0 - 1.5 - line < 3.0:
            continue
        n_run = int(((T[:, 0] >= line - 8.0) & (T[:, 0] > r.x + 2.0)).sum())
        if n_run == 0:
            continue
        dn = float(np.hypot(opp[:, 0] - r.x, opp[:, 1] - r.y).min())
        end_x = float(r.ex) if r.kind == "pass" and np.isfinite(r.ex) else -1.0
        rows.append((float(end_x >= line + 1.0), r.x / 10, abs(r.y - CY) / 10, min(dn, NEAR_CAP), min(n_run, 3), (line - r.x) / 10))
    from .fit import logistic_fit

    A = np.array(rows, float)
    w = logistic_fit(np.column_stack([np.ones(len(A)), A[:, 1:]]), A[:, 0])
    return {"g0": round(float(w[0]), 4), "g_x": round(float(w[1]), 4), "g_wide": round(float(w[2]), 4),
            "g_near": round(float(w[3]), 4), "g_run": round(float(w[4]), 4), "g_gap": round(float(w[5]), 4),
            "_n": int(len(A)), "_rate%": round(100 * float(A[:, 0].mean()), 2)}


def through_choice_data(frames, params=None, recv_max: float = 6.0):
    """身後球傳給哪一位：每次真實的身後球（與 fit_through_propensity 相同的狀態與定義：落點在防線之後至少 1 m）
    建立選項集合 = 防線附近的接球者（防線前後 8 m 內、在持球者前方 2 m 以上的可見隊友）；
    被選者 = 離「傳球路線（傳球者到落點的線段）」最近、且在 recv_max 公尺內的接球者。身後球送到接球者前方，
    以落點指派接球者會偏向比較深的接球者（係數隨指派方式大幅改變），以路線指派不依賴球送多深。
    特徵定義與引擎相同（decision.through_choice_features）。回傳 (X, 組別 g, 是否被選 y)。"""
    from ..config import load_params
    from ..engine.decision import lane_threats, through_choice_features

    pp = (params or load_params()).passing
    Xs, gs, ys = [], [], []
    g = 0
    for r in frames[frames.kind == "pass"].itertuples():
        if not np.isfinite([r.ex, r.ey]).all():
            continue
        f = s3.parse_frame(r.frame)
        O, K, T = f["O"], f["K"], f["T"]
        if len(O) < 3:
            continue
        line = max(float(O[:, 0].max()), 52.5)
        if r.x >= line or 105.0 - 1.5 - line < 3.0 or r.ex < line + 1.0:
            continue
        ok = (T[:, 0] >= line - 8.0) & (T[:, 0] > r.x + 2.0)
        if not ok.any():
            continue
        A = T[ok]
        v = np.array([r.ex - r.x, r.ey - r.y])
        t = np.clip(((A - (r.x, r.y)) @ v) / max(float(v @ v), 1e-9), 0.0, 1.0)
        de = np.hypot(*(A - ((r.x, r.y) + t[:, None] * v)).T)      # 接球者離「傳球路線」的垂直距離
        i = int(de.argmin())
        if de[i] > recv_max:
            continue
        opp = np.vstack([O, K]) if len(K) else O
        sep = np.sqrt(((A[:, None, :] - opp[None, :, :]) ** 2).sum(-1)).min(1)
        lane = lane_threats(r.x, r.y, A, opp, np.ones(len(opp)), pp).sum(1)
        Xs.append(through_choice_features(r.x, r.y, line, A, sep, lane))
        gs.append(np.full(len(A), g))
        y = np.zeros(len(A))
        y[i] = 1.0
        ys.append(y)
        g += 1
    return np.vstack(Xs), np.concatenate(gs), np.concatenate(ys)


def fit_through_choice(frames) -> dict:
    from ..engine.decision import THROUGH_CHOICE_FEATURES

    X, g, y = through_choice_data(frames)
    b, ll, base = fit_conditional_logit(X, g, y)
    out = {k: round(float(v), 4) for k, v in zip(THROUGH_CHOICE_FEATURES, b)}
    out.update({"_n_passes": int(g.max()) + 1, "_mean_options": round(len(g) / (int(g.max()) + 1), 2),
                "_loglik": round(ll, 4), "_loglik_uniform": round(base, 4)})
    return out


def fit_carry_choice(frames) -> dict:
    from ..engine.decision import CARRY_CHOICE_FEATURES

    X, g, y, st = carry_choice_data(frames)
    b, ll, base = fit_conditional_logit(X, g, y)
    out = {k: round(float(v), 4) for k, v in zip(CARRY_CHOICE_FEATURES, b)}
    out.update({"_n_carries": int(g.max()) + 1, "_n_states": st["states"], "_mean_options": round(len(g) / (int(g.max()) + 1), 2),
                "_loglik": round(ll, 4), "_loglik_uniform": round(base, 4)})
    return out


def save_fitted360(shot_prop: dict, cross_prop: dict, clear_prop: dict, pass_choice: dict,
                   pass_success: dict | None = None, pass_height: dict | None = None,
                   carry_prop: dict | None = None, carry_choice: dict | None = None,
                   through_prop: dict | None = None, through_choice: dict | None = None,
                   path: Path = CONFIG_DIR / "fitted360.yaml") -> Path:
    """360 資料擬合的模型；載入時在 fitted.yaml 之後覆蓋 engine_params.yaml 的同名項目"""
    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith("_")}
    doc = {"_source": SOURCE,
           "_samples": {"shot": shot_prop.get("_n"), "cross": cross_prop.get("_n"), "clear": clear_prop.get("_n"),
                        "pass_choice": pass_choice.get("_n_passes"),
                        "pass_success": (pass_success or {}).get("_n"), "carry": (carry_prop or {}).get("_n")},
           "_pass_choice_fit": {k: v for k, v in pass_choice.items() if k.startswith("_")},
           "_pass_success_fit": {k: v for k, v in (pass_success or {}).items() if k.startswith("_")},
           "_pass_height_fit": {k: v for k, v in (pass_height or {}).items() if k.startswith("_")},
           "shot_propensity": clean(shot_prop), "cross_propensity": clean(cross_prop),
           "clear_propensity": clean(clear_prop), "pass_choice": clean(pass_choice)}
    if pass_success:
        doc["pass_success"] = clean(pass_success)
    if pass_height:
        doc["pass_height"] = clean(pass_height)
    if carry_prop:
        doc["carry_propensity"] = clean(carry_prop)
    if through_prop:
        doc["_through_propensity_fit"] = {k: v for k, v in through_prop.items() if k.startswith("_")}
        doc["through_propensity"] = clean(through_prop)
    if through_choice:
        doc["_through_choice_fit"] = {k: v for k, v in through_choice.items() if k.startswith("_")}
        doc["through_choice"] = clean(through_choice)
    if carry_choice:
        doc["_carry_choice_fit"] = {k: v for k, v in carry_choice.items() if k.startswith("_")}
        doc["carry_choice"] = clean(carry_choice)
    path.write_text("# 由 `python -m footballsim statsbomb fit360` 產生，請勿手動修改\n"
                    + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ 傳球對象選擇模型（條件 logit）
def pass_choice_data(frames, params=None, recv_max: float = 10.0):
    """每次運動戰傳球（非傳中）建立選項集合：畫面內距傳球者 3–65 m 的隊友（含門將）。
    被選者 = 離傳球落點最近、且在 recv_max 公尺內的隊友（接球者常會再跑幾公尺去接）。
    畫面外的隊友不在選項中；條件 logit 在「只觀察到部分選項」時，係數仍一致（IIA）。
    只保留「找得到接球者」的傳球，會偏向短傳：長傳的接球者常在畫面外，失敗長傳的落點附近也常沒有隊友（擬合資料的
    平均傳球距離比全部真實傳球短 20%）。與傳球成功率的擬合相同，以「起點區域 × 傳球距離 × 高度」分層的反向抽樣權重
    校正（每層的保留比例的倒數，上限 10）。
    回傳 (特徵 X, 組別 g, 是否被選 y, 各組權重)，特徵定義與引擎相同（decision.pass_choice_features）。"""
    from ..config import load_params
    from ..engine.decision import lane_threats, pass_choice_features
    from ..engine.pitch import threat_for

    p = params or load_params()
    th = threat_for(p)
    pp = p.passing
    Xs, gs, ys = [], [], []
    g = 0
    n_all, n_keep, strata = np.zeros(36), np.zeros(36), []
    for r in frames[(frames.kind == "pass")].itertuples():
        if not np.isfinite([r.ex, r.ey]).all():
            continue
        st = ((int(np.digitize(r.x, [35.0, 70.0])) * 6 + int(np.digitize(np.hypot(r.ex - r.x, r.ey - r.y), _END_BINS)))
              * 2 + int(r.height in HEIGHT_LOFTED))
        n_all[st] += 1
        f = s3.parse_frame(r.frame)
        A = np.vstack([f["T"], f["G"]]) if len(f["G"]) else f["T"]
        gk = np.r_[np.zeros(len(f["T"]), bool), np.ones(len(f["G"]), bool)]
        O = np.vstack([f["O"], f["K"]]) if len(f["K"]) else f["O"]
        if len(A) < 2 or len(O) == 0:
            continue
        d = np.hypot(A[:, 0] - r.x, A[:, 1] - r.y)
        ok = (d >= 3.0) & (d <= 65.0)
        if ok.sum() < 2:
            continue
        A, gk = A[ok], gk[ok]
        de = np.hypot(A[:, 0] - r.ex, A[:, 1] - r.ey)
        i = int(de.argmin())
        if de[i] > recv_max:
            continue
        sep = np.sqrt(((A[:, None, :] - O[None, :, :]) ** 2).sum(-1)).min(1)
        lane = lane_threats(r.x, r.y, A, O, np.ones(len(O)), pp).sum(1)
        X = pass_choice_features(r.x, r.y, A, sep, lane, gk, th.xt_many(A) - th.xt_at(r.x, r.y))
        Xs.append(X)
        gs.append(np.full(len(A), g))
        y = np.zeros(len(A))
        y[i] = 1.0
        ys.append(y)
        n_keep[st] += 1
        strata.append(st)
        g += 1
    wg = np.minimum(n_all[strata] / np.maximum(n_keep[strata], 1), 10.0)
    return np.vstack(Xs), np.concatenate(gs), np.concatenate(ys), wg


def fit_conditional_logit(X: np.ndarray, g: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                          n_iter: int = 50, wg: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """條件 logit 的牛頓法：P(i | 組) ∝ exp(x_i·β)；wg 為各組權重（反向抽樣權重）。
    回傳 (β, 每組平均對數概似, 均勻選擇的基準)"""
    ng = int(g.max()) + 1
    wg = np.ones(ng) if wg is None else np.asarray(wg, float)
    wi = wg[g]
    b = np.zeros(X.shape[1])
    cnt = np.bincount(g, minlength=ng)
    for _ in range(n_iter):
        u = X @ b
        mx = np.full(ng, -np.inf)
        np.maximum.at(mx, g, u)
        e = np.exp(u - mx[g])
        P = e / np.bincount(g, weights=e, minlength=ng)[g]
        xbar = np.vstack([np.bincount(g, weights=P * X[:, k], minlength=ng) for k in range(X.shape[1])]).T
        Z = X - xbar[g]
        grad = ((y * wi)[:, None] * Z).sum(0) - l2 * b
        H = (Z * (P * wi)[:, None]).T @ Z + l2 * np.eye(len(b))
        step = np.linalg.solve(H, grad)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    u = X @ b
    mx = np.full(ng, -np.inf)
    np.maximum.at(mx, g, u)
    lse = mx + np.log(np.bincount(g, weights=np.exp(u - mx[g]), minlength=ng))
    ll = float(((u * y * wi).sum() - (wg * lse).sum()) / wg.sum())
    return b, ll, float(-(wg * np.log(cnt)).sum() / wg.sum())


def fit_pass_choice(frames) -> dict:
    from ..engine.decision import PASS_CHOICE_FEATURES

    X, g, y, wg = pass_choice_data(frames)
    b, ll, base = fit_conditional_logit(X, g, y, wg=wg)
    out = {k: round(float(v), 4) for k, v in zip(PASS_CHOICE_FEATURES, b)}
    out.update({"_n_passes": int(g.max()) + 1, "_mean_options": round(len(g) / (int(g.max()) + 1), 2),
                "_loglik": round(ll, 4), "_loglik_uniform": round(base, 4)})
    return out


# ------------------------------------------------------------------ 傳球成功率模型（依位置特徵）
HEIGHT_LOFTED = {"Low Pass", "High Pass"}
PASS_SUCCESS_FEATURES = ("s0", "s_d", "s_d2", "s_lane", "s_recv", "s_press", "s_back", "s_gk",
                         "s_loft", "s_loft_lane", "s_loft_d", "s_loft_recv")
_RING = np.array([[3, 0], [-3, 0], [0, 3], [0, -3], [2.1, 2.1], [-2.1, 2.1], [2.1, -2.1], [-2.1, -2.1]], float)
_END_BINS = (5.0, 10.0, 20.0, 30.0, 45.0)


def intended_receiver(B: np.ndarray, E: np.ndarray, mates: np.ndarray):
    """傳球的目標隊友（成功與失敗用同一規則）：在「傳球者 → 落點」射線附近的畫面內隊友，
    投影距離介於 0.4 × 落點距離 與 落點距離 + 15 m（球在 8 m 內就停下＝被擋，放寬到 45 m）、
    垂直偏離 ≤ max(3.5 m, 投影距離 × 0.25)；取「偏離 + 0.25 × |投影 − 落點距離|」最小者。
    失敗傳球的落點是被攔截或出界的位置，不是原本的目標，所以不能用「離落點最近的隊友」。回傳索引或 None。"""
    le = float(np.hypot(*(E - B)))
    if len(mates) == 0 or le < 1.0:
        return None
    u = (E - B) / le
    V = mates - B
    t = V @ u
    perp = np.abs(V[:, 0] * u[1] - V[:, 1] * u[0])
    t_hi = le + 15.0 if le >= 8.0 else 45.0
    cand = (t >= max(2.0, 0.4 * le)) & (t <= t_hi) & (perp <= np.maximum(3.5, 0.25 * t))
    if not cand.any():
        return None
    return int(np.argmin(np.where(cand, perp + 0.25 * np.abs(t - le), np.inf)))


def pass_success_data(frames, params=None):
    """運動戰腳下傳球（非傳中、非頭槌）的成功率特徵，定義與引擎相同（attributes 取平均）：
    傳球距離、路線攔截威脅（decision.lane_threats）、接球點的到達壓力（decision.arrival_pressure）、
    傳球者受壓（高斯核，σ = pressure.sigma）、回傳、傳給門將、高球（StatsBomb 的 Low/High Pass）。
    只保留目標隊友找得到、且其周圍 3 m 與傳球路線兩側 3 m 都在可視範圍內的傳球。
    這個篩選與結果有關（失敗傳球較常找不到目標），以「結果 × 區域 × 落點距離 × 高度」分層的
    反向抽樣權重校正（每層內的保留比例的倒數，上限 10）。回傳 (特徵 dict, 是否成功, 權重, 統計)。"""
    from ..config import load_params
    from ..engine.decision import arrival_pressure, lane_threats

    p = params or load_params()
    pp, sig_p, sig_r = p.passing, p.pressure.sigma, p.pressure.recv_sigma
    # 頭槌傳球不列入：引擎的頭球屬於爭頂解析，不是持球後的傳球決策
    ps = frames[(frames.kind == "pass") & (frames.body != "Head")]
    feats, allp = [], []
    for r in ps.itertuples():
        if not np.isfinite([r.ex, r.ey]).all():
            continue
        B, E = np.array([r.x, r.y], float), np.array([r.ex, r.ey], float)
        le = float(np.hypot(*(E - B)))
        loft = int(r.height in HEIGHT_LOFTED)
        allp.append([int(r.ok), r.x, le, loft, 0])
        f = s3.parse_frame(r.frame)
        mates = np.vstack([f["T"], f["G"]]) if len(f["G"]) else f["T"]
        j = intended_receiver(B, E, mates)
        if j is None:
            continue
        Rj = mates[j]
        d = float(np.hypot(*(Rj - B)))
        seg = B + np.linspace(0.15, 0.95, 6)[:, None] * (Rj - B)
        nrm = np.array([-(Rj - B)[1], (Rj - B)[0]]) / max(d, 1e-6)
        if not s3.inside(s3.parse_poly(r.vis), np.vstack([Rj + _RING, seg + 3 * nrm, seg - 3 * nrm])).all():
            continue
        O = np.vstack([f["O"], f["K"]]) if len(f["K"]) else f["O"]
        if len(O):
            w1 = np.ones(len(O))
            lane = float(lane_threats(B[0], B[1], Rj[None, :], O, w1, pp).sum())
            dr = np.hypot(O[:, 0] - Rj[0], O[:, 1] - Rj[1])[None, :]
            rpa = float(arrival_pressure(dr, np.array([d]), w1, pp, sig_r)[0])
            dp = np.hypot(O[:, 0] - B[0], O[:, 1] - B[1])
            press = float(np.exp(-dp * dp / (2 * sig_p * sig_p)).sum())
        else:
            lane = rpa = press = 0.0
        allp[-1][4] = 1
        feats.append((d, lane, rpa, press, float(Rj[0] - B[0] < -3.0), float(j >= len(f["T"])), loft, int(r.ok),
                      float(r.x)))
    F = np.array(feats, float)
    ALL = np.array(allp, float)

    def strata(ok, x, le, loft):
        return ((ok.astype(int) * 3 + np.digitize(x, [35.0, 70.0])) * 6 + np.digitize(le, _END_BINS)) * 2 + loft.astype(int)

    s_all = strata(ALL[:, 0], ALL[:, 1], ALL[:, 2], ALL[:, 3])
    kept = ALL[:, 4] == 1
    n_all = np.bincount(s_all, minlength=72)
    n_keep = np.bincount(s_all[kept], minlength=72)
    w = np.minimum(n_all[s_all[kept]] / np.maximum(n_keep[s_all[kept]], 1), 10.0)
    cols = {"dist": F[:, 0], "lane": F[:, 1], "rpa": F[:, 2], "P": F[:, 3], "back": F[:, 4], "gk": F[:, 5],
            "lofted": F[:, 6], "bx": F[:, 8]}
    info = {"_n": int(len(F)), "_n_all": int(len(ALL)), "_completion_all%": round(100 * float(ALL[:, 0].mean()), 1),
            "_completion_kept%": round(100 * float(F[:, 7].mean()), 1),
            "_completion_weighted%": round(100 * float(np.average(F[:, 7], weights=w)), 1)}
    return cols, F[:, 7], w, info


def pass_success_design(cols) -> np.ndarray:
    """特徵矩陣，欄位順序同 PASS_SUCCESS_FEATURES（與 decision.pass_success_logit 一致）"""
    d = cols["dist"] / 10.0
    sl, sr, sp = np.sqrt(cols["lane"]), np.sqrt(cols["rpa"]), np.sqrt(cols["P"])
    lo = cols["lofted"]
    return np.column_stack([np.ones_like(d), d, d * d, sl, sr, sp, cols["back"], cols["gk"],
                            lo, lo * sl, lo * d, lo * sr])


def weighted_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-4, n_iter: int = 100):
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-(X @ b)))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (w * (y - p)) - l2 * b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


PASS_HEIGHT_FEATURES = ("h0", "h_d", "h_d2", "h_lane", "h_press", "h_back", "h_gk", "h_own", "h_final")


def pass_height_design(cols) -> np.ndarray:
    """傳球高度選擇模型的特徵（與 decision.lofted_logit 一致）"""
    d = cols["dist"] / 10.0
    bx = cols["bx"]
    return np.column_stack([np.ones_like(d), d, d * d, np.sqrt(cols["lane"]), np.sqrt(cols["P"]), cols["back"],
                            cols["gk"], (bx < 35).astype(float), (bx >= 70).astype(float)])


def _logloss(X, y, w, b) -> float:
    p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-6, 1 - 1e-6)
    return round(float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w)), 4)


def fit_pass_success(frames, params=None) -> tuple[dict, dict]:
    """回傳 (傳球成功率模型, 傳球高度選擇模型)。
    高度是球員的技術選擇：真實比賽在本方三區約 35% 的腳下傳球是高球（含低平球），短傳也有約兩成；
    引擎若以期望價值挑高度，幾乎只會踢地面球（高球的成功率低很多），所以高度改用資料擬合的選擇機率。"""
    cols, y, w, info = pass_success_data(frames, params)
    X = pass_success_design(cols)
    b = weighted_logistic(X, y, w)
    out = {k: round(float(v), 4) for k, v in zip(PASS_SUCCESS_FEATURES, b)}
    out.update(info)
    out["_weighted_logloss"] = _logloss(X, y, w, b)
    Xh, yh = pass_height_design(cols), cols["lofted"]
    bh = weighted_logistic(Xh, yh, w)
    hgt = {k: round(float(v), 4) for k, v in zip(PASS_HEIGHT_FEATURES, bh)}
    hgt.update({"_n": info["_n"], "_lofted_weighted%": round(100 * float(np.average(yh, weights=w)), 1),
                "_weighted_logloss": _logloss(Xh, yh, w, bh)})
    return out, hgt
