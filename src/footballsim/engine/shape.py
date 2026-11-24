"""陣型快照：每個事件前計算 22 人的戰術座標（§4.4）

順序：防守區塊 → 進攻陣型 → 無球跑位（候選點取樣）→ 盯人追蹤 → 站位誤差與恍神 → 壓迫持球者。
回傳的座標都在「持球隊 T 的本隊座標系」（T 進攻方向 = +x）。
"""
from __future__ import annotations

import json
import math

import numpy as np

from ..config import CONFIG_DIR
from ..model.attributes import ATTR
from .pitch import BOX_HALF, BOX_X, CY, LENGTH, WIDTH

FK_DISTANCE = 9.15


def interp(levels, v: float) -> float:
    """1–5 級參數的線性內插（允許小數，例如心態 3.6）"""
    v = min(max(float(v), 1.0), 5.0)
    i = int(v)
    if i >= 5:
        return float(levels[4])
    f = v - i
    return float(levels[i - 1]) * (1 - f) + float(levels[i]) * f


def offside_line(pos: np.ndarray, d_slots) -> float:
    """倒數第二名防守者（含門將）的 x；本方半場不會越位"""
    if len(d_slots) < 2:
        return 52.5
    xs = np.sort(pos[d_slots, 0])
    return max(float(xs[-2]), 52.5)


def box_pool(sim):
    """禁區跑位資料（config/box_runs.yaml，由 360 資料產生）：(球 x 陣列, 球 |y−34| 陣列, 站位清單)"""
    pool = getattr(sim.p, "_box_pool", None)
    if pool is None:
        path = CONFIG_DIR / sim.p.shape.box_run_file
        bxs, bdys, spots = [], [], []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("- ["):
                    a, b, sp = json.loads(line[2:])
                    bxs.append(a)
                    bdys.append(b)
                    spots.append(np.asarray(sp, float).reshape(-1, 2))
        pool = (np.asarray(bxs, float), np.asarray(bdys, float), spots)
        sim.p._box_pool = pool
    return pool


def _box_runs(sim, p, pos, movers, bx, by, rng) -> set:
    """球在前場時，從真實畫面中找持球位置相近者，讓禁區＋前沿（x ≥ box_zone_x、|y−34| ≤ box_zone_half）
    的攻方人數與站位照那個真實畫面：已在區域內的人優先佔真實站位，不夠再由最前面的隊友補上；
    沒分到站位的人退到區域邊緣（真實畫面裡那裡沒有那麼多人）。
    同一段進攻沿用同一組站位（只要它仍是相近畫面之一），避免每個事件都換位置。
    站位不在這裡判斷越位：防守者盯人後防線會再後退，越位最後以實際防線判斷（compute 的 6b）。
    回傳被分配站位的球員。"""
    assigned: set = set()
    bxs, bdys, spots = box_pool(sim)
    if not len(bxs):
        return assigned
    d2 = ((bxs - bx) / p.box_run_sx) ** 2 + ((bdys - abs(by - CY)) / p.box_run_sy) ** 2
    k = min(int(p.box_run_k), len(bxs))
    near = np.argpartition(d2, k - 1)[:k]
    prev = getattr(sim, "_box_cfg", None)
    idx = prev if prev is not None and prev in near else int(rng.choice(near))
    sim._box_cfg = idx
    cfg = spots[idx]
    in_zone = [s for s in movers if pos[s, 0] >= p.box_zone_x and abs(pos[s, 1] - CY) <= p.box_zone_half]
    if len(cfg):
        side = 1.0 if by >= CY else -1.0
        tgt = np.column_stack([np.minimum(cfg[:, 0], LENGTH - 1.0), CY + cfg[:, 1] * side])
        outside = [movers[i] for i in np.argsort(-pos[movers, 0]) if movers[i] not in in_zone]
        free = in_zone + outside[: max(len(cfg) + 1 - len(in_zone), 0)]
        for t in tgt[np.argsort(-tgt[:, 0])]:
            if not free:
                break
            j = min(free, key=lambda s: (pos[s, 0] - t[0]) ** 2 + (pos[s, 1] - t[1]) ** 2)
            pos[j] = t
            free.remove(j)
            assigned.add(j)
    for s in in_zone:
        if s not in assigned:
            pos[s, 0] = p.box_zone_x - 0.5
    return assigned


def _att_ranks(p, pos, mv, bx: float, x_line: float, rng) -> None:
    """持球方的縱向層次（360 資料）：最前面的幾名球員分布在「球 → 對方防線」之間的固定比例位置
    （中位數：球在中場時約 0.95、0.84、0.71、0.54、0.37、0.15；進入前場後後兩人退到球附近）。
    誰在前、誰在後與左右位置仍由陣型決定（依陣型模板的前後排序），這裡只決定前後距離。"""
    span = x_line - bx
    if not mv or span <= 1.0:
        return
    f = min(max((bx - p.att_ranks_x[0]) / (p.att_ranks_x[1] - p.att_ranks_x[0]), 0.0), 1.0)
    fr = (1 - f) * np.asarray(p.att_ranks_mid, float) + f * np.asarray(p.att_ranks_final, float)
    # 陣型同深度的兩人（例如兩名邊鋒）以小幅隨機決定誰較前
    order = np.argsort(-(pos[mv, 0] + rng.normal(0.0, 0.5, len(mv))))
    for k, i in enumerate(order[: len(fr)]):
        pos[mv[i], 0] = bx + fr[k] * span


def compute(sim, T: int, bx: float, by: float, carrier, set_piece: bool = False):
    p = sim.p.shape
    pos = sim.pos
    rng = sim.rng_shape
    C = sim.C
    D = 1 - T
    tD, tT = sim.teams[D], sim.teams[T]
    d_slots = sim.active_slots(D)
    t_slots = sim.active_slots(T)
    gk_d, gk_t = sim.gk_slot(D), sim.gk_slot(T)
    d_out = [s for s in d_slots if s != gk_d]
    t_out = [s for s in t_slots if s != gk_t]

    # ---- 1. 防守區塊（在防守方自己的座標系計算後翻轉）----
    dbx, dby = LENGTH - bx, WIDTH - by
    base = interp(p.line_base, tD.tactics.line_height + 0.5 * tD.mentality_offset)
    follow = p.line_follow_back if dbx < 52.5 else p.line_follow_fwd
    back = base + follow * (dbx - 52.5)
    # 退守的下限（360）：防線隨球快速後退到禁區線附近後就守住，不再隨球退進禁區
    back = min(max(back, p.line_floor0 + p.line_floor_k * dbx, 4.0), 55.0)
    # 疲勞只在能量低於門檻後才讓陣型拉長（比賽後段的空檔）
    fat = max(0.0, p.fatigue_e0 - float(sim.energy[d_out].mean())) if d_out else 0.0
    length = interp(p.block_length, tD.tactics.press) * (1.0 + p.fatigue_stretch * fat)
    # 球接近本方球門時整個區塊往後壓縮（中場退到禁區前沿）
    length *= 1.0 - p.block_compress * max(0.0, 1.0 - dbx / p.compress_dist)
    if d_out:
        ks = np.asarray(d_out) - 11 * D
        oop = tD.oop[ks]
        gx = back + oop[:, 0] * length
        # 球越接近本方禁區，防線越往中路收縮
        narrow = p.def_width * (1.0 - p.def_narrow * max(0.0, 1.0 - dbx / 40.0))
        gy = CY + (oop[:, 1] - CY) * narrow + p.def_shift * (dby - CY)
        pos[d_out, 0] = LENGTH - gx
        pos[d_out, 1] = WIDTH - gy
    if gk_d is not None:
        gx = min(p.gk_x_min + p.gk_x_ratio * dbx, p.gk_x_max)
        pos[gk_d, 0] = LENGTH - gx
        pos[gk_d, 1] = WIDTH - (CY + 0.15 * (dby - CY))
    off0 = offside_line(pos, d_slots)

    # ---- 2. 進攻陣型（最前線不超過越位線，除非球已在線後）----
    m = tT.tactics.mentality - 3 + tT.mentality_offset
    x_max = min(104.0, max(off0, bx)) - 0.3
    front = min(x_max, bx + p.att_front_ahead + p.mentality_front * m)
    # 後衛線：球後 att_back_behind，但組織進攻時不會退到門前（360：球在本方三區時最深的外野球員約在 9–15 m）
    floor = min(p.att_back_floor, max(bx, p.att_back_min))
    back_a = min(max(bx - p.att_back_behind + p.mentality_back * m, floor), max(front - 18.0, 4.0))
    width = interp(p.att_width, tT.tactics.width)
    if t_out:
        ks = np.asarray(t_out) - 11 * T
        ip = tT.ip[ks]
        xr = ip[:, 0] + (tT.ip_adj[ks] if bx > 70 else 0.0)
        pos[t_out, 0] = back_a + np.minimum(xr, 1.0) * (front - back_a)
        pos[t_out, 1] = CY + (ip[:, 1] - CY) * width + p.att_shift * (by - CY)
        _att_ranks(p, pos, [s for s in t_out if s != carrier], bx, x_max, rng)
    if gk_t is not None:
        pos[gk_t, 0] = min(max(2.0 + 0.22 * bx, 2.0), 28.0)
        pos[gk_t, 1] = CY + 0.1 * (by - CY)

    # ---- 3. 無球跑位：抽 K 個候選點，選「xT + 開闊度」最好者；K = 1 + 無球跑位//20 ----
    movers = [s for s in t_out if s != carrier]
    # 最前線（陣型位置貼近越位線）的球員負責壓住對方防線：只沿防線橫向找位置，也不回頭接應
    line_runner = np.zeros(len(movers), dtype=bool)
    if movers:
        line_runner = pos[movers, 0] >= x_max - p.front_band
    if movers and d_slots:
        n = len(movers)
        tmpl = pos[movers].copy()
        # 候選點數 = 1 + 無球跑位/20，小數部分以機率進位（避免 59→3、60→4 的階梯效應）
        kf = 1.0 + sim.eff[movers, ATTR.OFF_THE_BALL] / 20.0
        K = np.minimum(np.floor(kf).astype(int) + (rng.random(n) < kf % 1.0), 6)
        step = rng.normal(0.0, p.move_sigma, (n, 6, 2))
        step[line_runner, :, 0] *= p.front_sigma_x
        cand = tmpl[:, None, :] + step
        np.clip(cand[..., 0], 1.0, max(x_max, 1.5), out=cand[..., 0])
        np.clip(cand[..., 1], 1.0, WIDTH - 1.0, out=cand[..., 1])
        diff = cand[:, :, None, :] - pos[d_slots][None, None, :, :]
        dmin = np.sqrt((diff * diff).sum(-1)).min(-1)
        drift = np.sqrt(((cand - tmpl[:, None, :]) ** 2).sum(-1))
        score = (sim.threat.xt_many(cand) + p.move_open_w * np.minimum(dmin, 8.0)
                 - p.move_drift_w * drift)
        score[np.arange(6)[None, :] >= K[:, None]] = -np.inf
        best = score.argmax(1)
        pos[movers] = cand[np.arange(n), best]

    # ---- 3b. 接應：最近的隊友往持球者靠近，提供短傳選項（團隊配合越高，接應越積極）----
    if carrier is not None and movers and p.support_n > 0:
        cx, cy = bx, by
        dv = pos[movers] - (cx, cy)
        dd = np.sqrt((dv * dv).sum(1))
        for i in [i for i in np.argsort(dd) if not line_runner[i]][: int(p.support_n)]:
            d = float(dd[i])
            if d <= p.support_dist:
                continue
            s = movers[i]
            frac = p.support_frac * (0.6 + 0.4 * sim.eff[s, ATTR.TEAMWORK] / 100)
            move = (d - p.support_dist) * frac
            pos[s] -= dv[i] / d * move
            pos[s, 0] = min(pos[s, 0], x_max)

    # ---- 3c. 禁區跑位（360 資料）：球在前場時，禁區與前沿的攻方站位取自真實比賽中相近持球位置的畫面 ----
    runners: set = set()
    if carrier is not None and movers and bx >= p.box_run_min_x:
        runners = _box_runs(sim, p, pos, movers, bx, by, rng)
    else:
        sim._box_cfg = None

    # ---- 4. 盯人：由最深的防守者開始，追蹤半徑內最危險的進攻者 ----
    mark_of: dict = {}
    if d_out and movers:
        apos = pos[movers]
        # 危險度 = 位置威脅 + 離球越近越優先（區域防守往球側收縮，接應者也會被盯）
        dball = np.hypot(apos[:, 0] - bx, apos[:, 1] - by)
        danger = sim.threat.xt_many(apos) + p.mark_ball_w * np.exp(-dball / p.mark_ball_scale)
        dpos = pos[d_out]
        dist = np.sqrt(((dpos[:, None, :] - apos[None, :, :]) ** 2).sum(-1))
        assigned = np.zeros(len(movers), dtype=bool)
        marking = sim.comp[d_out, C.MARKING]          # mark_of：防守者 → 所盯進攻者離球的距離
        # 盯人範圍：球在對方半場（我方高位逼搶）時採人盯人式壓迫，範圍較大；退守時以區域為主
        radius = p.mark_radius + p.mark_radius_press * max(0.0, (dbx - 52.5) / 52.5)
        for i in np.argsort(-dpos[:, 0]):
            ok = (~assigned) & (dist[i] < radius)
            if not ok.any():
                continue
            j = int(np.argmax(np.where(ok, danger, -1.0)))
            assigned[j] = True
            mark_of[d_out[i]] = float(dball[j])
            ax, ay = apos[j]
            vx, vy = LENGTH - ax, CY - ay
            norm = math.hypot(vx, vy) or 1.0
            # 盯人距離：防守者盯人能力 vs 進攻者無球跑位（擺脫）
            dm = (p.mark_dist_min + p.mark_dist_span * (1.0 - marking[i] / 100.0)
                  + p.mark_otb * (sim.eff[movers[j], ATTR.OFF_THE_BALL] - sim.R) / 10)
            # 盯人鬆緊（360 資料）：離球越近貼得越緊，離球遠的人主要由區域站位照顧；禁區內一律緊盯
            f_ball = math.exp(-float(dball[j]) / p.mark_ball_scale)
            w = p.mark_weight + (p.mark_weight_ball - p.mark_weight) * f_ball
            if ax >= BOX_X and abs(ay - CY) <= BOX_HALF:
                dm *= p.mark_dist_box
                w = max(w, p.mark_weight_box)
            s = d_out[i]
            pos[s, 0] += w * (ax + vx / norm * dm - pos[s, 0])
            pos[s, 1] += w * (ay + vy / norm * dm - pos[s, 1])

    # ---- 5. 防守站位誤差（站位、專注、團隊配合）與恍神 ----
    if d_out:
        q = sim.comp[d_out, C.DEF_QUALITY] / 100.0
        sig = p.def_err_min + p.def_err_span * (1.0 - q)
        # 防守組織隨體能下降逐步鬆動（開賽最嚴密）
        sig = sig * (1.0 + p.def_err_fatigue * (1.0 - sim.energy[d_out]))
        conc = sim.eff[d_out, ATTR.CONCENTRATION]
        p_lapse = p.lapse_base * (1.5 - conc / 100.0) * (1.0 + p.lapse_fatigue * np.maximum(0.0, p.fatigue_e0 - sim.energy[d_out]))
        lapse = rng.random(len(d_out)) < p_lapse
        sig = sig * np.where(lapse, p.lapse_mult, 1.0)
        pos[d_out] += rng.normal(0.0, 1.0, (len(d_out), 2)) * sig[:, None]

    # ---- 6. 壓迫持球者（定位球時改為退到 9.15 m 外）----
    if carrier is not None and d_out:
        dv = pos[d_out] - (bx, by)
        dd = np.sqrt((dv * dv).sum(1))
        if set_piece:
            close = dd < FK_DISTANCE
            if close.any():
                scale = FK_DISTANCE / np.maximum(dd[close], 0.1)
                pos[np.asarray(d_out)[close]] = np.array([bx, by]) + dv[close] * scale[:, None]
        else:
            # 最近的防守者站到「球與球門之間」貼身（jockey）；第二人在其身後補位。
            # 壓迫強度與壓迫能力越高，貼得越近；在本方危險區域一律上前封堵。
            # 比分情境：落後方加強前場壓迫，領先方退守
            press = tD.tactics.press + p.mentality_press * tD.mentality_offset
            engaged = dbx <= max(interp(p.press_engage, press), p.always_engage)
            phi0 = interp(p.press_phi, press) if engaged else p.press_phi_passive
            gxv, gyv = LENGTH - bx, CY - by
            gn = math.hypot(gxv, gyv) or 1.0
            ux, uy = gxv / gn, gyv / gn
            # 不逼搶時（球離本方球門夠遠、壓迫戰術低）不派人上前，持球者可以很空（360：本方三區約三分之一的傳球
            # 最近對手在 6 m 外）；逼搶時一人貼身、一人補位，另依壓迫戰術派人貼住離球最近的接應點（切斷傳球路線）
            pressers = np.argsort(dd)[:2] if (engaged or p.press_passive) else np.zeros(0, dtype=int)
            for rank, i in enumerate(pressers):
                s = d_out[i]
                # 只有一人上搶；補位的第二人若正盯著球附近的進攻者，就留在原盯人位置（不放空接應點）
                if rank == 1 and mark_of.get(s, np.inf) < p.cover_keep_mark:
                    continue
                phi = (phi0 + 0.004 * (sim.comp[s, C.PRESS] - sim.R)) * (0.7 + 0.3 * sim.energy[s])
                phi = min(max(phi, 0.0), 0.8)
                if rank == 0:
                    dist_t = p.jockey_min + p.jockey_span * (1.0 - phi)
                    if bx >= BOX_X and abs(by - CY) <= BOX_HALF:   # 禁區內貼得更近
                        dist_t *= p.jockey_box
                    # 站在側面、引導對手往外，而不是正擋在射門路線上
                    lat = -p.jockey_lat if by >= CY else p.jockey_lat
                else:
                    dist_t = p.jockey_min + p.jockey_span * (1.0 - phi) + p.cover_dist
                    lat = p.cover_lat if pos[s, 1] > by else -p.cover_lat
                tx = bx + ux * min(dist_t, gn * 0.9) - uy * lat
                ty = by + uy * min(dist_t, gn * 0.9) + ux * lat
                dx, dy = tx - pos[s, 0], ty - pos[s, 1]
                need = math.hypot(dx, dy)
                if need < 1e-6:
                    continue
                reach = (p.press_reach_base + p.press_reach_phi * phi) * (1.0 if rank == 0 else 0.7)
                frac = min(1.0, reach / need)
                pos[s, 0] += dx * frac
                pos[s, 1] += dy * frac
            n_extra = int(round(interp(p.press_extra, press))) if engaged else 0
            if n_extra > 0 and movers:
                used = {d_out[i] for i in pressers}
                opts_a = sorted(movers, key=lambda a: (pos[a, 0] - bx) ** 2 + (pos[a, 1] - by) ** 2)
                done = 0
                for a in opts_a[: n_extra + 2]:
                    if done >= n_extra:
                        break
                    free_d = [k for k in d_out if k not in used]
                    if not free_d:
                        break
                    k = min(free_d, key=lambda q: (pos[q, 0] - pos[a, 0]) ** 2 + (pos[q, 1] - pos[a, 1]) ** 2)
                    ax, ay = pos[a]
                    vb = np.array([bx - ax, by - ay])
                    nb = float(np.hypot(*vb)) or 1.0
                    # 站在接應者與球之間（切斷傳球路線）
                    tx, ty = ax + vb[0] / nb * p.press_cut_dist, ay + vb[1] / nb * p.press_cut_dist
                    dx, dy = tx - pos[k, 0], ty - pos[k, 1]
                    need = math.hypot(dx, dy)
                    phi = min(max((phi0 + 0.004 * (sim.comp[k, C.PRESS] - sim.R)) * (0.7 + 0.3 * sim.energy[k]), 0.0), 0.8)
                    reach = p.press_reach_base + p.press_reach_phi * phi
                    if need > reach + p.press_extra_range:
                        continue
                    frac = min(1.0, reach / max(need, 1e-6))
                    pos[k, 0] += dx * frac
                    pos[k, 1] += dy * frac
                    used.add(k)
                    done += 1

    # ---- 6b. 以防守站位完成後的實際防線判斷越位：壓線前鋒跟著防線移動，其餘攻方球員不得越位 ----
    if movers and d_slots:
        x_lim = min(104.0, max(offside_line(pos, d_slots), bx)) - 0.3
        mv = np.asarray(movers)
        follow = line_runner & ~np.isin(mv, list(runners))
        pos[mv[follow], 0] += x_lim - x_max
        pos[mv, 0] = np.minimum(pos[mv, 0], x_lim)

    np.clip(pos[:, 0], 0.5, LENGTH - 0.5, out=pos[:, 0])
    np.clip(pos[:, 1], 0.5, WIDTH - 0.5, out=pos[:, 1])

    # ---- 7. 位置連續性：上一個快照到現在，每人最多只能跑「速度 × 經過時間」----
    # 這讓攻防轉換有時間差：失球方需要回防時間，反擊與身後球因此成立。
    dt = sim.elapsed - sim.last_shape_elapsed
    if sim.gpos is not None and not set_piece and dt < p.free_reposition_s:
        prev = sim.gpos.copy()
        if T == 1:
            prev[:, 0] = LENGTH - prev[:, 0]
            prev[:, 1] = WIDTH - prev[:, 1]
        dt = max(dt, p.min_move_dt)
        speed = sim.vmax * p.move_speed_frac * (0.6 + 0.4 * sim.energy)
        dslots = 11 * D
        speed[dslots:dslots + 11] *= 0.75 + 0.25 * sim.eff[dslots:dslots + 11, ATTR.WORK_RATE] / 100
        delta = pos - prev
        dist = np.sqrt((delta * delta).sum(1))
        scale = np.minimum(1.0, speed * dt / np.maximum(dist, 1e-9))
        mask = sim.active.copy()
        if carrier is not None:
            mask[carrier] = False
        pos[mask] = prev[mask] + delta[mask] * scale[mask, None]

    if carrier is not None:
        pos[carrier] = (bx, by)
    sim.gpos = pos.copy() if T == 0 else np.column_stack((LENGTH - pos[:, 0], WIDTH - pos[:, 1]))
    sim.last_shape_elapsed = sim.elapsed
    return pos, offside_line(pos, d_slots)
