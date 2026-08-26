"""Grammar-first Coulomb-diamond detection, current mode (detection track).

Analysis only: takes a measured current map I(Vg, Vsd) with its voltage axes.
No simulation code here -- see synth_current_mode.py for the synthetic model
used by the golden tests.

Input convention
----------------
    img : 2D array, shape (n_vsd, n_vg)
    vg  : 1D array of gate voltages, uniformly spaced (either direction)
    vsd : 1D array of bias voltages, uniformly spaced, spanning through ~0
          (the dead band / apex line must be inside the scan)

Axes in either direction are accepted; the data is oriented internally to
vg ascending along columns and vsd DESCENDING down rows (top row = +Vsd).

Output: DetectResult (see bottom). All voltages in the units of the axes.

Port of the JS reference implementation (detect_current_mode.js); the golden
tests in test_golden.py are the contract. Zero deps beyond numpy.
"""

from dataclasses import dataclass, field, asdict
import math
import numpy as np


# ---------------------------------------------------------------- geometry

class Grid:
    """Uniform measurement grid. Rows are Vsd (descending), cols Vg (ascending)."""

    def __init__(self, vg, vsd):
        vg = np.asarray(vg, dtype=float)
        vsd = np.asarray(vsd, dtype=float)
        if vg.ndim != 1 or vsd.ndim != 1:
            raise ValueError("vg and vsd must be 1D axes")
        for name, ax in (("vg", vg), ("vsd", vsd)):
            d = np.diff(ax)
            if not (np.all(d > 0) or np.all(d < 0)):
                raise ValueError(f"{name} axis must be strictly monotonic")
            if np.ptp(np.abs(d)) > 1e-6 * np.abs(d).mean():
                raise ValueError(f"{name} axis must be uniformly spaced")
        self.vg = vg
        self.vsd = vsd                      # descending after orientation
        self.Nx = len(vg)
        self.Ny = len(vsd)
        self.sx = float(vg[1] - vg[0])      # > 0
        self.sy = float(vsd[0] - vsd[1])    # > 0 (descending rows)

    def col_vg(self, i):
        return self.vg[0] + np.asarray(i, dtype=float) * self.sx

    def row_vsd(self, j):
        return self.vsd[0] - np.asarray(j, dtype=float) * self.sy


def orient(img, vg, vsd):
    """Return (img, Grid) oriented to vg ascending / vsd descending."""
    img = np.asarray(img, dtype=float)
    vg = np.asarray(vg, dtype=float)
    vsd = np.asarray(vsd, dtype=float)
    if img.shape != (len(vsd), len(vg)):
        raise ValueError(f"img shape {img.shape} != (len(vsd), len(vg)) = "
                         f"({len(vsd)}, {len(vg)})")
    if vg[1] < vg[0]:
        vg = vg[::-1].copy(); img = img[:, ::-1]
    if vsd[1] > vsd[0]:
        vsd = vsd[::-1].copy(); img = img[::-1, :]
    return np.ascontiguousarray(img), Grid(vg, vsd)


# ---------------------------------------------------------------- config

@dataclass
class DetectConfig:
    """Every data-unit constant of the detector.

    Defaults are tuned for scans with gate span ~1 V, bias span ~0.24 V and
    apex spacings ~0.1-0.3 V (the golden-test regime). For different scales,
    the members marked [scale with dVg] should be scaled with your expected
    addition voltage, and [scale with px] with your pixel pitch.
    """
    # expected slope-sign convention: family '+' has dVsd/dVg > 0, '-' < 0
    min_abs_slope: float = 0.04          # grammar sign guard on refits
    slope_scan_lo: float = -1.15         # ladder-constrained '-' scan range
    slope_scan_hi: float = -0.18
    fallback_m_plus: float = 0.35        # used only when a channel is empty
    fallback_m_minus: float = -0.55

    # evidence / tensor (pixel-unit; [scale with px] if resolution differs a lot)
    blur_fine_px: int = 1
    blur_coarse_px: int = 2              # applied twice
    tensor_blur_px: int = 2              # applied twice
    tipzone_blur_px: int = 4
    wedge_deg: float = 4.5               # horizontal nuisance wedge
    mag_gate_nsigma: float = 5.0         # MAD multiples over median |dI/dVg|
    tipzone_weight: float = 0.12

    # dead band
    dead_thr_nsigma: float = 2.2
    dead_persist_rows: int = 8
    dead_persist_frac: float = 0.6
    dead_onset_rows: int = 4
    dead_onset_cap_rows: int = 10
    dead_max: float = 0.05               # [scale with bias span]

    # region track
    region_thr_nsigma: float = 4.0
    region_min_wann: float = 0.12

    # intercept combs
    comb_bins: int = 280
    comb_sigma_bins: float = 1.3
    comb_w_diff: float = 0.65
    comb_w_region: float = 0.35
    comb_contrast_topk: int = 30
    peak_rel_thr: float = 0.22
    peak_abs_frac: float = 0.14
    agree_comb: float = 0.5              # cos(2*dBeta) gates
    agree_refit: float = 0.55
    soft_sign_weight: float = 0.35
    scan_sign_weight: float = 0.3
    scan_tooth_halfwidth: float = 0.009  # [scale with dVg]
    scan_steps: int = 90
    scan_adopt_ratio: float = 0.85

    # refits / pooling
    refit_band_dc: float = 3.0           # in comb-bin widths
    min_points: int = 10
    strong_frac: float = 0.35
    min_support_vg: float = 0.035        # [scale with dVg]
    censor_margin: float = 0.006         # [scale with bias span]
    dedupe_dc: float = 0.012             # [scale with dVg]
    dedupe_dm: float = 0.12

    # vertices / delta / ladder
    vertex_margin_vg: float = 0.025      # [scale with dVg]
    vertex_band_margin: float = 0.03     # extrapolation allowance addend
    delta_cluster: float = 0.006         # [scale with bias span]
    apex_gate_lo: float = 0.008
    apex_gate_hi: float = 0.02
    apex_gate_nsigma: float = 2.5
    apex_merge_vg: float = 0.02          # [scale with dVg]
    apex_cluster_vg: float = 0.03        # [scale with dVg]
    apex_nms_vg: float = 0.07            # just under the min physical gap [scale with dVg]
    gap_resolved_lo: float = 0.38        # x median: admits bimodal (even/odd) spectra
    gap_resolved_hi: float = 2.3

    # width collapse
    wc_cap_frac: float = 1.15
    wc_center_frac: float = 0.35
    wc_center_pad: float = 0.01          # [scale with dVg]
    wc_width_pad: float = 0.015          # [scale with dVg]
    wc_max_miss: int = 3
    wc_min_points: int = 4
    wc_min_width: float = 0.01           # [scale with dVg]


# ------------------------------------------------------------- primitives

def _pctile(arr, q):
    """Exact replica of the JS pctileArr: sorted[floor(q*n)], clamped."""
    a = np.sort(np.asarray(arr, dtype=float).ravel())
    n = len(a)
    return a[max(0, min(n - 1, int(math.floor(q * n))))]


def _median_js(arr):
    """JS `sorted[n >> 1]` median convention."""
    a = np.sort(np.asarray(arr, dtype=float).ravel())
    return a[len(a) >> 1]


def _box_blur(src, r):
    """Separable mean filter over in-bounds samples only (edge-renormalized)."""
    if r <= 0:
        return src.copy()
    Ny, Nx = src.shape

    def _axis(a, r, axis):
        n = a.shape[axis]
        cs = np.cumsum(a, axis=axis)
        cs = np.concatenate([np.zeros_like(np.take(cs, [0], axis=axis)), cs], axis=axis)
        idx = np.arange(n)
        lo = np.clip(idx - r, 0, n)
        hi = np.clip(idx + r + 1, 0, n)
        s = np.take(cs, hi, axis=axis) - np.take(cs, lo, axis=axis)
        cnt = (hi - lo).astype(float)
        shape = [1, 1]; shape[axis] = n
        return s / cnt.reshape(shape)

    return _axis(_axis(src, r, 1), r, 0)


def _sobel_pair(src):
    """3x3 Sobel /8; border ring left at zero (matches the JS reference)."""
    Ny, Nx = src.shape
    gx = np.zeros_like(src)
    gy = np.zeros_like(src)
    c = src
    gx[1:-1, 1:-1] = (c[:-2, 2:] + 2 * c[1:-1, 2:] + c[2:, 2:]
                      - c[:-2, :-2] - 2 * c[1:-1, :-2] - c[2:, :-2]) / 8.0
    gy[1:-1, 1:-1] = (c[2:, :-2] + 2 * c[2:, 1:-1] + c[2:, 2:]
                      - c[:-2, :-2] - 2 * c[:-2, 1:-1] - c[:-2, 2:]) / 8.0
    return gx, gy


# --------------------------------------------- layer 1: row conditioning

def condition_rows(img):
    """Per-row offset from the blockade-floor low quantile; interpolated where
    a row has no zero reference. Display/region-track conditioning only: the
    differential evidence is immune to row offsets by construction."""
    Ny, Nx = img.shape
    srt = np.sort(img, axis=1)
    q = lambda p: srt[:, max(0, min(Nx - 1, int(math.floor(p * Nx))))]
    est = q(0.12).copy()
    spread = q(0.30) - q(0.05)
    lo, hi = _pctile(est, 0.2), _pctile(est, 0.85)
    split = (lo + hi) / 2
    bimodal = (hi - lo) > 0.15
    med_spread = _pctile(spread, 0.5)
    reliable = (est < split) if bimodal else (spread < 3.0 * med_spread + 1e-4)

    est_s = est.copy()
    for j in range(Ny):
        if not reliable[j]:
            continue
        w = [est[jj] for jj in range(max(0, j - 2), min(Ny, j + 3)) if reliable[jj]]
        w.sort()
        est_s[j] = w[len(w) >> 1]

    off = np.zeros(Ny)
    rel_idx = np.flatnonzero(reliable)
    for j in range(Ny):
        if reliable[j]:
            off[j] = est_s[j]
            continue
        prev = rel_idx[rel_idx < j]
        nxt = rel_idx[rel_idx > j]
        if len(prev) and len(nxt):
            a, b = prev[-1], nxt[0]
            off[j] = est_s[a] + (est_s[b] - est_s[a]) * (j - a) / (b - a)
        elif len(prev):
            off[j] = est_s[prev[-1]]
        elif len(nxt):
            off[j] = est_s[nxt[0]]
    return img - off[:, None], off, reliable


# ------------------------------------------------- dead band + noise floor

def estimate_dead_band(img_c, grid, cfg):
    """Dead band as a changepoint on the smoothed row-power profile, with a
    persistence test against clutter spikes and onset extrapolation back along
    the ramp. Recenters img_c IN PLACE on the dead-band median (the dead band
    is the pure-noise calibration sample). Returns (Vdead, sigma)."""
    Ny, Nx = img_c.shape
    # zero-bias row, JS d2pY(0) convention: round((vsd_edge - 0)/sy) with the top
    # EDGE at vsd[0] + sy/2 and round-half-up (matters exactly at the tie row)
    j0 = int(math.floor(float(grid.vsd[0]) / grid.sy + 0.5 + 0.5))
    j0 = max(0, min(Ny - 1, j0))
    cen = img_c[max(0, j0 - 2):min(Ny, j0 + 3), :]
    med0 = _pctile(cen, 0.5)
    sig0 = max(1e-4, 1.4826 * _pctile(np.abs(cen - med0), 0.5))

    srt = np.sort(img_c - med0, axis=1)
    P0 = srt[:, max(0, min(Nx - 1, int(math.floor(0.9 * Nx))))]
    P = np.empty(Ny)
    for j in range(Ny):
        a, b = max(0, j - 1), min(Ny, j + 2)
        P[j] = P0[a:b].mean()
    thr = cfg.dead_thr_nsigma * sig0

    def persists(j, d):
        above = n = 0
        for t in range(cfg.dead_persist_rows):
            jj = j + d * t
            if jj < 1 or jj > Ny - 2:
                break
            n += 1
            if P[jj] > thr:
                above += 1
        return n > 0 and above / n >= cfg.dead_persist_frac

    jhi = jlo = j0
    while jhi < Ny - 2 and not (P[jhi] > thr and persists(jhi, +1)):
        jhi += 1
    while jlo > 1 and not (P[jlo] > thr and persists(jlo, -1)):
        jlo -= 1

    def onset(jc, d):
        s = n = 0
        for t in range(cfg.dead_onset_rows):
            j = jc + d * t
            if j + d < 1 or j + d > Ny - 2:
                break
            s += (P[j + d] - P[j]) * d
            n += 1
        slope = s / n if n else 0.0
        back = min(cfg.dead_onset_cap_rows, P[jc] / slope) if slope > 1e-9 else 0.0
        return abs(float(grid.row_vsd(jc))) - back * grid.sy

    vdead = min(cfg.dead_max, max(1.5 * grid.sy, (onset(jhi, +1) + onset(jlo, -1)) / 2))
    pool = img_c[min(jlo + 1, j0):max(jhi - 1, j0) + 1, :].copy()  # copy BEFORE
    med = _pctile(pool, 0.5)                                       # in-place recenter
    img_c -= med
    sigma = max(1e-4, 1.4826 * _pctile(np.abs(pool - med), 0.5))
    return vdead, sigma


# ---------------------------------------- layers 2-3: dual evidence tracks

def differential_evidence(img_c, grid, vdead, cfg):
    """Pure dI/dVg at two scales (per-pixel max, scale-normalized), sign kept.
    Exactly immune to per-row additive structure; the image's dI/dVsd (carrier
    of the A'(Vsd) amplitude-growth nuisance) never enters. Returns dict with
    gx (signed, per pixel dI/dVg in data units/px-normalized), mag, wAnn."""
    s1 = _box_blur(img_c, cfg.blur_fine_px)
    s2 = _box_blur(_box_blur(img_c, cfg.blur_coarse_px), cfg.blur_coarse_px)
    g1x, _ = _sobel_pair(s1)
    g2x, _ = _sobel_pair(s2)
    m1 = 1.3 * np.abs(g1x)
    m2 = 2.6 * np.abs(g2x)
    use1 = m1 >= m2
    gx = np.where(use1, g1x, g2x)
    mag = np.where(use1, m1, m2)

    Ny, Nx = img_c.shape
    srt = np.sort(mag, axis=1)
    w_row = srt[:, max(0, min(Nx - 1, int(math.floor(0.95 * Nx))))]
    w_norm = _pctile(w_row, 0.97) + 1e-9
    w_ann = np.minimum(1.0, w_row / w_norm)
    av = np.abs(grid.row_vsd(np.arange(Ny)))
    w_ann[av < vdead + 2 * grid.sy] = 0.0
    ws = w_ann.copy()
    ws[1:-1] = (w_ann[:-2] + w_ann[1:-1] + w_ann[2:]) / 3.0
    return {"gx": gx, "mag": mag, "wAnn": ws}


def structure_tensor(ev, grid, cfg):
    """RIDGE MODE: structure tensor of the |dI/dVg| map's own gradients, so the
    current image's vertical derivative never touches orientation. Family
    channels split by sign(gx)*sign(Vsd) BEFORE smoothing."""
    Ny, Nx = ev["mag"].shape
    mag, gx, w_ann = ev["mag"], ev["gx"], ev["wAnn"]
    med = _median_js(mag)
    thr = med + cfg.mag_gate_nsigma * 1.4826 * _median_js(np.abs(mag - med))
    rgx, rgy = _sobel_pair(mag)

    sv = np.sign(grid.row_vsd(np.arange(Ny)))[:, None]
    sgx = np.sign(gx)
    gated = (sgx != 0) & (sv != 0) & (mag > thr)
    fam_plus = gated & (sgx * sv < 0)
    fam_minus = gated & (sgx * sv > 0)

    out = {}
    near = {}
    for key, sel in (("+", fam_plus), ("-", fam_minus)):
        near[key] = _box_blur(sel.astype(float), cfg.tipzone_blur_px)
    tip_zone = np.where((near["+"] > 0.03) & (near["-"] > 0.03), cfg.tipzone_weight, 1.0)

    wedge = cfg.wedge_deg * math.pi / 180.0
    border = np.zeros((Ny, Nx), dtype=bool)
    border[3:Ny - 3, 3:Nx - 3] = True

    for key, sel in (("+", fam_plus), ("-", fam_minus)):
        Jxx = _box_blur(_box_blur(np.where(sel, rgx * rgx, 0.0), cfg.tensor_blur_px), cfg.tensor_blur_px)
        Jyy = _box_blur(_box_blur(np.where(sel, rgy * rgy, 0.0), cfg.tensor_blur_px), cfg.tensor_blur_px)
        Jxy = _box_blur(_box_blur(np.where(sel, rgx * rgy, 0.0), cfg.tensor_blur_px), cfg.tensor_blur_px)
        E = Jxx + Jyy
        coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy * Jxy) / (E + 1e-14)
        phi = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
        th = phi + math.pi / 2
        m = (-np.sin(th) * grid.sy) / (np.cos(th) * grid.sx + 1e-15)
        beta = np.arctan(m)
        ok = border & (E >= 1e-14) & (w_ann[:, None] > 0) & (np.abs(beta) >= wedge) & (mag > thr)
        wgt = np.where(ok, coh ** 3 * np.sqrt(E) * w_ann[:, None] * tip_zone, 0.0)
        out[key] = {"beta": beta, "wgt": wgt}
    return out


# ------------------------------------------------- stage A: orientation

def orientation_histogram(st):
    beta = st["beta"].ravel()
    wgt = st["wgt"].ravel()
    sel = wgt > 0
    b = np.floor(beta[sel] * 180.0 / math.pi + 90.0 + 0.5).astype(int) % 180  # JS round
    hist = np.bincount(b, weights=wgt[sel], minlength=180).astype(float)
    K = np.array([0.054, 0.244, 0.403, 0.244, 0.054])
    idx = (np.arange(180)[:, None] + np.arange(-2, 3)[None, :]) % 180
    return (hist[idx] * K[None, :]).sum(axis=1)


def find_family_candidates(hist, expect_positive, cfg, n_cand=3):
    """Up to n_cand local maxima in the family half-plane; Stage A proposes,
    Stage B (comb contrast) disposes."""
    NB = len(hist)
    h = hist.copy()
    beta_axis = np.arange(NB) - 90
    mask = (beta_axis >= 5) if expect_positive else (beta_axis <= -5)
    out = []
    for _ in range(n_cand):
        hm = np.where(mask, h, -np.inf)
        bi = int(np.argmax(hm))
        bv = hm[bi]
        if not np.isfinite(bv) or bv <= 0 or (out and bv < 0.25 * out[0]["height"]):
            break
        ks = np.arange(-5, 6)
        bb = (bi + ks) % NB
        sw = hist[bb].sum()
        swx = (hist[bb] * ks).sum()
        center = bi + swx / (sw + 1e-12)
        beta_deg = center - 90.0
        out.append({"betaDeg": beta_deg, "height": float(bv),
                    "m": math.tan(beta_deg * math.pi / 180.0)})
        h[(bi + np.arange(-8, 9)) % NB] = 0.0
    if not out:
        out.append({"betaDeg": 19.3 if expect_positive else -28.8, "height": 0.0,
                    "m": cfg.fallback_m_plus if expect_positive else cfg.fallback_m_minus,
                    "fallback": True})
    return out


# ------------------------------------------- region track + intercept combs

def region_track(img_c, grid, vdead, sigma, ev, cfg):
    """Per-row blockade intervals (threshold n*sigma, subpixel endpoints) =
    diamond cross-sections. Endpoints labeled by which family's edge they are."""
    Ny, Nx = img_c.shape
    thr = cfg.region_thr_nsigma * sigma
    endpoints = {"+": [], "-": []}
    intervals = []
    for j in range(2, Ny - 2):
        v = float(grid.row_vsd(j))
        if abs(v) < vdead + 2 * grid.sy or ev["wAnn"][j] < cfg.region_min_wann:
            continue
        mask = img_c[j] > thr
        i = 0
        while i < Nx:
            if mask[i]:
                i += 1
                continue
            start = i
            while i < Nx and not mask[i]:
                i += 1
            end = i
            ln = end - start
            if start >= 2 and end <= Nx - 2 and 2 <= ln <= Nx * 0.5 and mask[start - 1] and mask[end]:
                def sub(a, b):
                    va, vb = img_c[j, a], img_c[j, b]
                    t = (thr - va) / (vb - va) if abs(vb - va) > 1e-12 else 0.5
                    return float(grid.col_vg(a + min(1.0, max(0.0, t)) + 0.5))
                vg_l = sub(start - 1, start)
                vg_r = sub(end - 1, end)
                w = float(ev["wAnn"][j])
                endpoints["+" if v > 0 else "-"].append({"vg": vg_l, "v": v, "w": w})
                endpoints["-" if v > 0 else "+"].append({"vg": vg_r, "v": v, "w": w})
                intervals.append({"v": v, "center": (vg_l + vg_r) / 2,
                                  "width": vg_r - vg_l, "w": w})
    return {"endpoints": endpoints, "intervals": intervals, "thr": thr}


def _gated_pixels(ev, st, fam, grid, cfg):
    """(vg, vs, beta, mag, sgn_ok) arrays for one family channel, border-trimmed."""
    Ny, Nx = ev["mag"].shape
    wgt, beta = st["wgt"], st["beta"]
    sel = np.zeros((Ny, Nx), dtype=bool)
    sel[3:Ny - 3, 3:Nx - 3] = True
    sel &= (wgt > 0) & (ev["mag"] > 0) & (ev["wAnn"][:, None] > 0)
    jj, ii = np.nonzero(sel)
    v = grid.row_vsd(jj)
    exp_sign = (-1 if fam == "+" else 1) * np.sign(v)
    sgn_ok = np.sign(ev["gx"][jj, ii]) == exp_sign
    return {"vg": grid.col_vg(ii), "vs": grid.row_vsd(jj), "beta": beta[jj, ii],
            "mag": ev["mag"][jj, ii], "wann": ev["wAnn"][jj], "sgn_ok": sgn_ok}


def intercept_profile(px, fam, m, beta_fam, region, grid, cfg):
    """Slope-constrained fused intercept comb (differential + region tracks)."""
    c_min = grid.vsd[-1] - max(0.0, m) * grid.vg[-1] - 0.02
    c_max = grid.vsd[0] - min(0.0, m) * grid.vg[-1] + 0.02
    NB = cfg.comb_bins
    dc = (c_max - c_min) / NB
    ag = np.maximum(0.0, np.cos(2 * (px["beta"] - beta_fam)))
    keep = ag >= cfg.agree_comb
    sgn_w = np.where(px["sgn_ok"], 1.0, cfg.soft_sign_weight)
    c = px["vs"] - m * px["vg"]
    b = np.floor((c - c_min) / dc).astype(int)
    ok = keep & (b >= 0) & (b < NB)
    dif = np.bincount(b[ok], weights=(px["mag"] * ag ** 3 * sgn_w * px["wann"])[ok],
                      minlength=NB).astype(float)
    reg = np.zeros(NB)
    for e in region["endpoints"][fam]:
        bb = int(math.floor((e["v"] - m * e["vg"] - c_min) / dc))
        if 0 <= bb < NB:
            reg[bb] += e["w"]

    ks = np.arange(-4, 5)
    K = np.exp(-ks ** 2 / (2 * cfg.comb_sigma_bins ** 2))
    K /= K.sum()

    def smooth(a):
        out = np.zeros(NB)
        for k, w in zip(ks, K):
            lo, hi = max(0, k), min(NB, NB + k)
            out[max(0, -k):min(NB, NB - k)] += w * a[lo:hi]
        return out

    dif_s, reg_s = smooth(dif), smooth(reg)
    d_hi = dif_s.max() + 1e-12
    r_hi = reg_s.max() + 1e-12
    fused = cfg.comb_w_diff * dif_s / d_hi + cfg.comb_w_region * reg_s / r_hi
    return {"fused": fused, "dif": dif_s, "reg": reg_s, "cMin": c_min,
            "cMax": c_max, "dc": dc}


def comb_contrast(prof, cfg):
    a = np.sort(prof["fused"])[::-1]
    return a[:cfg.comb_contrast_topk].sum() / (a.sum() + 1e-12)


def find_profile_peaks(prof, cfg):
    f = prof["fused"]
    NB = len(f)
    srt = np.sort(f)
    med, mx = srt[NB >> 1], srt[-1]
    thr = max(med + cfg.peak_rel_thr * (mx - med), cfg.peak_abs_frac * mx)
    peaks = []
    for b in range(2, NB - 2):
        if (f[b] > thr and f[b] >= f[b - 1] and f[b] > f[b + 1]
                and f[b] >= f[b - 2] and f[b] > f[b + 2]):
            d = 0.5 * (f[b - 1] - f[b + 1]) / (f[b - 1] - 2 * f[b] + f[b + 1] + 1e-12)
            peaks.append({"c": prof["cMin"] + (b + d) * prof["dc"],
                          "height": float(f[b]), "bin": b})
    peaks.sort(key=lambda p: -p["height"])
    kept = []
    for p in peaks:
        if not any(abs(q["bin"] - p["bin"]) < 5 for q in kept):
            kept.append(p)
    kept.sort(key=lambda p: p["c"])
    return kept


def ladder_constrained_slope(px, apexes, cfg):
    """Matched filter for the weak family: score each candidate slope by the
    coincidence of its intercept comb with the strong family's apex skeleton."""
    if len(px["vg"]) < 40 or len(apexes) < 2:
        return None
    sgn_w = np.where(px["sgn_ok"], 1.0, cfg.scan_sign_weight)
    w = px["mag"] * sgn_w * px["wann"]
    apx = np.asarray(apexes)
    best = None
    for t in range(cfg.scan_steps + 1):
        m = cfg.slope_scan_lo + (cfg.slope_scan_hi - cfg.slope_scan_lo) * t / cfg.scan_steps
        b_f = math.atan(m)
        ag = np.cos(2 * (px["beta"] - b_f))
        keep = ag >= 0.5
        c = px["vs"][keep] - m * px["vg"][keep]
        d = np.abs(c[:, None] - (-m * apx)[None, :])
        dmin = d.min(axis=1)
        hit = dmin < cfg.scan_tooth_halfwidth
        s = float((w[keep][hit] * ag[keep][hit] ** 2
                   * (1 - dmin[hit] / cfg.scan_tooth_halfwidth)).sum())
        if best is None or s > best["s"]:
            best = {"m": m, "s": s}
    return best


# ----------------------------------- stage C: refits + hierarchical pooling

def refine_line(px, fam, m0, beta_fam, c0, dc, vdead, cfg):
    ag = np.maximum(0.0, np.cos(2 * (px["beta"] - beta_fam)))
    keep = (ag >= cfg.agree_refit) & (np.abs(px["vs"] - m0 * px["vg"] - c0)
                                      <= cfg.refit_band_dc * dc)
    if keep.sum() < cfg.min_points:
        return None
    vg, vs = px["vg"][keep], px["vs"][keep]
    sgn_w = np.where(px["sgn_ok"][keep], 1.0, cfg.soft_sign_weight)
    w = px["mag"][keep] * ag[keep] ** 2 * sgn_w * px["wann"][keep]
    Sw, Sx, Sy = w.sum(), (w * vg).sum(), (w * vs).sum()
    Sxx, Sxy = (w * vg * vg).sum(), (w * vg * vs).sum()
    if Sw <= 0:
        return None
    D = Sw * Sxx - Sx * Sx
    if abs(D) < 1e-12:
        return None
    m = (Sw * Sxy - Sx * Sy) / D
    c = (Sxx * Sy - Sx * Sxy) / D
    rss = (w * (vs - m * vg - c) ** 2).sum()
    neff = Sw * Sw / (w * w).sum()
    s2 = rss / max(1.0, Sw * (1 - 2 / max(3.0, neff)))
    var_m, var_c = s2 * Sw / D, s2 * Sxx / D
    if fam == "+" and m < cfg.min_abs_slope:
        return None
    if fam == "-" and m > -cfg.min_abs_slope:
        return None
    strong = w > cfg.strong_frac * w.max()
    vg_lo, vg_hi = float(vg[strong].min()), float(vg[strong].max())
    if vg_hi - vg_lo < cfg.min_support_vg:
        return None
    inner = vdead
    cen = lambda vse: abs(vse) < inner + cfg.censor_margin
    return {"fam": fam, "m": float(m), "c": float(c),
            "seM": math.sqrt(max(var_m, 1e-12)), "seC": math.sqrt(max(var_c, 1e-12)),
            "vgLo": vg_lo, "vgHi": vg_hi,
            "censLo": cen(m * vg_lo + c), "censHi": cen(m * vg_hi + c),
            "nSupport": int(keep.sum())}


def dedupe_lines(lines, cfg):
    ok = sorted([l for l in lines if l], key=lambda l: l["c"])
    out = []
    for l in ok:
        if out and abs(l["c"] - out[-1]["c"]) < cfg.dedupe_dc \
                and abs(l["m"] - out[-1]["m"]) < cfg.dedupe_dm:
            if l["nSupport"] > out[-1]["nSupport"]:
                out[-1] = l
        else:
            out.append(l)
    return out


def shrink_family(lines, mode):
    ok = [l for l in lines if l]
    if not ok:
        return {"M": float("nan"), "tau": float("nan")}
    ws = np.array([1 / (l["seM"] ** 2) for l in ok])
    ms = np.array([l["m"] for l in ok])
    M = float((ws * ms).sum() / ws.sum())
    num = (ws * ((ms - M) ** 2 - 1 / ws)).sum()
    tau2 = max(0.0, num / ws.sum())
    tau = math.sqrt(tau2)
    for l, wl in zip(ok, ws):
        if mode == "raw":
            l["mShrunk"] = l["m"]
        elif mode == "full":
            l["mShrunk"] = M
        else:
            wg = 1 / tau2 if tau2 > 1e-12 else 1e12
            l["mShrunk"] = (wl * l["m"] + wg * M) / (wl + wg)
        l["cShrunk"] = l["c"]
    return {"M": M, "tau": tau}


# ------------------------- stage D: vertices; delta from apex collinearity

def build_vertices(plus, minus, grid, vdead, cfg):
    inner = vdead
    verts = []
    for a in plus:
        for b in minus:
            dm = a["mShrunk"] - b["mShrunk"]
            if abs(dm) < 1e-6:
                continue
            vg = (b["cShrunk"] - a["cShrunk"]) / dm
            vs = a["mShrunk"] * vg + a["cShrunk"]
            if not (grid.vg[0] - 0.02 <= vg <= grid.vg[-1] + 0.02):
                continue
            if not (grid.vsd[-1] - 0.01 <= vs <= grid.vsd[0] + 0.01):
                continue
            in_band = abs(vs) < inner + 0.01
            marg_a = cfg.vertex_margin_vg + ((inner / abs(a["mShrunk"]) + cfg.vertex_band_margin)
                                             if (in_band or a["censLo"] or a["censHi"]) else 0.0)
            marg_b = cfg.vertex_margin_vg + ((inner / abs(b["mShrunk"]) + cfg.vertex_band_margin)
                                             if (in_band or b["censLo"] or b["censHi"]) else 0.0)
            if vg < a["vgLo"] - marg_a or vg > a["vgHi"] + marg_a:
                continue
            if vg < b["vgLo"] - marg_b or vg > b["vgHi"] + marg_b:
                continue
            D = 1 / dm
            dvg = np.array([-vg * D, -D, vg * D, D])
            dvs = np.array([a["mShrunk"] * dvg[0] + vg, a["mShrunk"] * dvg[1] + 1,
                            a["mShrunk"] * dvg[2], a["mShrunk"] * dvg[3]])
            vvar = np.array([a["seM"] ** 2, a["seC"] ** 2, b["seM"] ** 2, b["seC"] ** 2])
            verts.append({"vg": float(vg), "vs": float(vs),
                          "Sxx": float((dvg * dvg * vvar).sum()),
                          "Syy": float((dvs * dvs * vvar).sum()),
                          "Sxy": float((dvg * dvs * vvar).sum())})
    # delta: apexes are collinear on Vsd = delta and delta lives INSIDE the dead
    # band; covariance-weighted center of the tightest candidate cluster
    cap = max(0.006, vdead)
    cand = [{"vs": v["vs"], "w": 1 / (v["Syy"] + 1e-8)}
            for v in verts if abs(v["vs"]) < cap]
    delta = 0.0
    if cand:
        best = None
        for c in cand:
            cl = [o for o in cand if abs(o["vs"] - c["vs"]) < cfg.delta_cluster]
            W = sum(o["w"] for o in cl)
            if best is None or W > best[0]:
                best = (W, cl)
        delta = sum(o["w"] * o["vs"] for o in best[1]) / best[0]
    for v in verts:
        gate = max(cfg.apex_gate_lo,
                   min(cfg.apex_gate_hi, cfg.apex_gate_nsigma * math.sqrt(max(0.0, v["Syy"]))))
        v["kind"] = "apex" if abs(v["vs"] - delta) < gate else "tip"
    return verts, delta


# ------------------------------------------------ stage E: ladder assembly

def assemble_ladder(verts, delta, lines, grid, cfg):
    apexes = sorted([v for v in verts if v["kind"] == "apex"], key=lambda v: v["vg"])
    merged = []
    for v in apexes:
        if merged and abs(v["vg"] - merged[-1]["vg"]) < cfg.apex_merge_vg:
            last = merged[-1]
            w1, w2 = 1 / (last["Sxx"] + 1e-12), 1 / (v["Sxx"] + 1e-12)
            last["vg"] = (w1 * last["vg"] + w2 * v["vg"]) / (w1 + w2)
            last["vs"] = (w1 * last["vs"] + w2 * v["vs"]) / (w1 + w2)
            last["Sxx"] = 1 / (w1 + w2)
        else:
            merged.append(dict(v))
    tips = [v for v in verts if v["kind"] == "tip"]
    diamonds = []
    for i in range(len(merged) - 1):
        aL, aR = merged[i], merged[i + 1]
        inside = [t for t in tips if aL["vg"] + 0.008 < t["vg"] < aR["vg"] - 0.008]
        ups = sorted([t for t in inside if t["vs"] > delta], key=lambda t: t["vs"])
        los = sorted([t for t in inside if t["vs"] < delta], key=lambda t: -t["vs"])
        diamonds.append({"aL": aL, "aR": aR,
                         "tipUp": ups[0] if ups else None,
                         "tipLo": los[0] if los else None})

    # apex COMB from line delta-crossings (robust to lost vertices), then
    # support-weighted clustering + NMS just under the minimum physical gap
    cross = sorted([{"x": (delta - l["c"]) / l["m"], "w": l.get("nSupport", 1)}
                    for l in lines if l and abs(l["m"]) > 0.05
                    and grid.vg[0] - 0.05 < (delta - l["c"]) / l["m"] < grid.vg[-1] + 0.05],
                   key=lambda c: c["x"])
    clus = []
    for c in cross:
        if clus and c["x"] - clus[-1]["x"] < cfg.apex_cluster_vg:
            last = clus[-1]
            last["x"] = (last["x"] * last["w"] + c["x"] * c["w"]) / (last["w"] + c["w"])
            last["w"] += c["w"]
            last["n"] += 1
        else:
            clus.append({"x": c["x"], "w": c["w"], "n": 1})
    clus.sort(key=lambda c: -c["w"])
    pos = []
    for c in clus:
        if not any(abs(p["x"] - c["x"]) < cfg.apex_nms_vg for p in pos):
            pos.append(c)
    pos.sort(key=lambda p: p["x"])

    spac = [pos[i + 1]["x"] - pos[i]["x"] for i in range(len(pos) - 1)]
    # uneven spacing is PHYSICS (disorder, even/odd spin pairing): gaps are
    # reported per diamond, never regularized onto a lattice; only gaps far
    # outside any physical mode are flagged unresolved
    gaps, dvg = [], float("nan")
    if spac:
        med = sorted(spac)[len(spac) >> 1]
        gaps = [{"vgL": pos[i]["x"], "vgR": pos[i + 1]["x"], "dVg": s,
                 "resolved": cfg.gap_resolved_lo * med < s < cfg.gap_resolved_hi * med}
                for i, s in enumerate(spac)]
        ok = sorted(g["dVg"] for g in gaps if g["resolved"])
        if ok:
            dvg = ok[len(ok) >> 1]

    heights = []
    for d in diamonds:
        if d["tipUp"]:
            heights.append(abs(d["tipUp"]["vs"] - delta))
        if d["tipLo"]:
            heights.append(abs(d["tipLo"]["vs"] - delta))
    ec_tips = float(np.mean(heights)) if heights else float("nan")
    return {"apexes": merged, "diamonds": diamonds, "dVg": dvg,
            "gaps": gaps, "EcTips": ec_tips}


# ------------------- width collapse: region-track E_C consistency check

def width_collapse(intervals, ladder, delta, cfg):
    tips, fits = [], []
    for d in ladder["diamonds"]:
        for side in (1, -1):
            w_cap = cfg.wc_cap_frac * (d["aR"]["vg"] - d["aL"]["vg"])
            cand = sorted([iv for iv in intervals
                           if d["aL"]["vg"] < iv["center"] < d["aR"]["vg"]
                           and np.sign(iv["v"] - delta) == side
                           and cfg.wc_min_width < iv["width"] < w_cap],
                          key=lambda iv: abs(iv["v"] - delta))
            # monotone envelope: convex-diamond cross-sections are NESTED and
            # SHRINKING in |Vsd|; three consecutive misses = tip passed
            run_c = (d["aL"]["vg"] + d["aR"]["vg"]) / 2
            last_w = d["aR"]["vg"] - d["aL"]["vg"]
            miss = 0
            pts = []
            for p in cand:
                if (abs(p["center"] - run_c) < cfg.wc_center_frac * last_w + cfg.wc_center_pad
                        and p["width"] < last_w + cfg.wc_width_pad):
                    pts.append(p)
                    run_c = 0.5 * (run_c + p["center"])
                    last_w = min(last_w, p["width"])
                    miss = 0
                elif pts:
                    miss += 1
                    if miss >= cfg.wc_max_miss:
                        break
            if len(pts) < cfg.wc_min_points:
                continue
            x = np.array([abs(p["v"] - delta) for p in pts])
            y = np.array([p["width"] for p in pts])
            w = np.array([p["w"] for p in pts])
            Sw, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
            Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
            D = Sw * Sxx - Sx * Sx
            if abs(D) < 1e-12:
                continue
            B = (Sw * Sxy - Sx * Sy) / D
            A = (Sxx * Sy - Sx * Sxy) / D
            if B < 0 and A > 0:
                tip = -A / B
                if 0 < tip < 0.12:
                    tips.append(tip)
                    fits.append({"side": side, "A": A, "B": B, "tip": tip})
    ec = float(np.mean(tips)) if tips else float("nan")
    return {"Ec": ec, "fits": fits}


# ---------------------------------------------------------- full pipeline

@dataclass
class DetectResult:
    Mp: float; Mm: float
    MpStageA: float; MmStageA: float
    alpha: float
    tauP: float; tauM: float
    delta: float
    Vdead: float
    sigma: float
    Ec: float
    EcPerDiamond: list
    EcTips: float
    EcRegion: float
    minus_via_ladder: bool
    lines_plus: list; lines_minus: list
    verts: list
    ladder: dict
    region: dict
    hist_plus: np.ndarray; hist_minus: np.ndarray
    img_conditioned: np.ndarray
    row_offsets: np.ndarray
    grid: Grid

    def summary(self):
        pd = " ".join(f"{1e3*g['Ec']:.0f}" for g in self.EcPerDiamond
                      if g["resolved"] and np.isfinite(g["Ec"]))
        return (f"M+ {self.Mp:.3f}  M- {self.Mm:.3f}  alpha {self.alpha:.3f}"
                f"{' (ladder scan)' if self.minus_via_ladder else ''}  |  "
                f"delta {1e3*self.delta:.2f} mV  Vdead {1e3*self.Vdead:.1f} mV  |  "
                f"Ec median {1e3*self.Ec:.1f} mV  per-diamond [{pd}] mV  "
                f"region {1e3*self.EcRegion:.1f} mV  |  "
                f"lines +{len(self.lines_plus)}/-{len(self.lines_minus)}  "
                f"verts {len(self.verts)}")


def _alpha(mp, mm):
    return (mp * abs(mm)) / (mp + abs(mm))


def run_detection(img, vg, vsd, cfg=None, shrink="hier"):
    """Full detection pipeline on a measured current map.

    Parameters
    ----------
    img : (n_vsd, n_vg) current map
    vg, vsd : voltage axes (any direction; vsd must span through ~0)
    cfg : DetectConfig, thresholds in data units (see class docstring)
    shrink : 'raw' | 'hier' | 'full' slope-pooling mode
    """
    cfg = cfg or DetectConfig()
    img, grid = orient(img, vg, vsd)
    img_c, row_off, _ = condition_rows(img)
    vdead, sigma = estimate_dead_band(img_c, grid, cfg)
    ev = differential_evidence(img_c, grid, vdead, cfg)
    st = structure_tensor(ev, grid, cfg)
    region = region_track(img_c, grid, vdead, sigma, ev, cfg)
    hist_p = orientation_histogram(st["+"])
    hist_m = orientation_histogram(st["-"])
    px_p = _gated_pixels(ev, st["+"], "+", grid, cfg)
    px_m = _gated_pixels(ev, st["-"], "-", grid, cfg)

    def pick(hist, px, fam, expect_pos):
        best = None
        for c in find_family_candidates(hist, expect_pos, cfg):
            prof = intercept_profile(px, fam, c["m"], c["betaDeg"] * math.pi / 180,
                                     region, grid, cfg)
            score = comb_contrast(prof, cfg) * max(c["height"], 1e-9) ** 0.25
            if best is None or score > best["score"]:
                best = {"cand": c, "prof": prof, "score": score}
        return best

    b_p = pick(hist_p, px_p, "+", True)
    fam_p, prof_p = b_p["cand"], b_p["prof"]

    # strong-family-first: apex skeleton from '+' pass-1 lines, then the '-'
    # slope by ladder-constrained scan (fall back to histogram + contrast)
    peaks_p0 = find_profile_peaks(prof_p, cfg)
    lines_p0 = [refine_line(px_p, "+", fam_p["m"], fam_p["betaDeg"] * math.pi / 180,
                            p["c"], prof_p["dc"], vdead, cfg) for p in peaks_p0]
    apex_skel = [-l["c"] / l["m"] for l in lines_p0 if l
                 and grid.vg[0] - 0.03 < -l["c"] / l["m"] < grid.vg[-1] + 0.03]
    b_m = pick(hist_m, px_m, "-", False)
    fam_m, prof_m = b_m["cand"], b_m["prof"]
    via_ladder = False
    if len(apex_skel) >= 3:
        scan = ladder_constrained_slope(px_m, apex_skel, cfg)
        if scan:
            prof_scan = intercept_profile(px_m, "-", scan["m"], math.atan(scan["m"]),
                                          region, grid, cfg)
            if comb_contrast(prof_scan, cfg) > cfg.scan_adopt_ratio * comb_contrast(prof_m, cfg):
                fam_m = {"betaDeg": math.degrees(math.atan(scan["m"])),
                         "height": fam_m["height"], "m": scan["m"]}
                prof_m = prof_scan
                via_ladder = True

    def pass_lines(px, fam, m, beta_deg, prof):
        return [refine_line(px, fam, m, beta_deg * math.pi / 180, p["c"],
                            prof["dc"], vdead, cfg)
                for p in find_profile_peaks(prof, cfg)]

    shr_p1 = shrink_family(pass_lines(px_p, "+", fam_p["m"], fam_p["betaDeg"], prof_p), "hier")
    shr_m1 = shrink_family(pass_lines(px_m, "-", fam_m["m"], fam_m["betaDeg"], prof_m), "hier")
    m_p2 = shr_p1["M"] if math.isfinite(shr_p1["M"]) else fam_p["m"]
    m_m2 = shr_m1["M"] if math.isfinite(shr_m1["M"]) else fam_m["m"]

    prof_p2 = intercept_profile(px_p, "+", m_p2, math.atan(m_p2), region, grid, cfg)
    prof_m2 = intercept_profile(px_m, "-", m_m2, math.atan(m_m2), region, grid, cfg)
    lines_p = dedupe_lines(pass_lines(px_p, "+", m_p2, math.degrees(math.atan(m_p2)), prof_p2), cfg)
    lines_m = dedupe_lines(pass_lines(px_m, "-", m_m2, math.degrees(math.atan(m_m2)), prof_m2), cfg)
    shr_p = shrink_family(lines_p, shrink)
    shr_m = shrink_family(lines_m, shrink)

    verts, delta = build_vertices(lines_p, lines_m, grid, vdead, cfg)
    ladder = assemble_ladder(verts, delta, lines_p + lines_m, grid, cfg)
    wc = width_collapse(region["intervals"], ladder, delta, cfg)

    al = _alpha(shr_p["M"], shr_m["M"])
    return DetectResult(
        Mp=shr_p["M"], Mm=shr_m["M"],
        MpStageA=fam_p["m"], MmStageA=fam_m["m"],
        alpha=al, tauP=shr_p["tau"], tauM=shr_m["tau"],
        delta=delta, Vdead=vdead, sigma=sigma,
        Ec=al * ladder["dVg"] if math.isfinite(ladder["dVg"]) else float("nan"),
        EcPerDiamond=[{"vgL": g["vgL"], "vgR": g["vgR"], "resolved": g["resolved"],
                       "Ec": al * g["dVg"] if g["resolved"] else float("nan")}
                      for g in ladder["gaps"]],
        EcTips=ladder["EcTips"], EcRegion=wc["Ec"],
        minus_via_ladder=via_ladder,
        lines_plus=lines_p, lines_minus=lines_m, verts=verts, ladder=ladder,
        region=region, hist_plus=hist_p, hist_minus=hist_m,
        img_conditioned=img_c, row_offsets=row_off, grid=grid)
