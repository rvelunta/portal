"""Synthetic Coulomb-diamond current maps (simulation model), current mode.

Kept strictly separate from the analysis (detect_current_mode.py): this module
only PRODUCES (img, vg, vsd, truth) tuples compatible with run_detection.

Exact port of the JS reference synthesis, including the mulberry32 RNG and
Marsaglia-polar normals, so a given (params, seed) reproduces the same
instance as the interactive demo.

Noise model:
- transport dead band |Vsd| < Vdead gating diamonds AND clutter (no transport,
  no current), with a linear current ramp above onset
- edge width w0 + smear * |Vsd - delta|  (bias-linear Coulomb-peak smearing)
- AR(1) row offsets (rho = 0.93), white pixel noise
- clutter ridges with random position/angle/amplitude
- faint excited-state step parallel to each mp edge
- apex spacings: disorder scatter (spacing_scatter) and/or even-odd
  alternation (spacing_ratio) -- addition energy is physics, not a lattice
"""

from dataclasses import dataclass
import math
import numpy as np

_MASK = 0xFFFFFFFF


class Mulberry32:
    """Bit-exact port of the JS demo RNG (mulberry32 + Marsaglia polar)."""

    def __init__(self, seed):
        self.a = seed & _MASK
        self.spare = None

    def u(self):
        self.a = (self.a + 0x6D2B79F5) & _MASK
        t = ((self.a ^ (self.a >> 15)) * ((1 | self.a) & _MASK)) & _MASK
        t = ((t + ((t ^ (t >> 7)) * ((61 | t) & _MASK) & _MASK)) ^ t) & _MASK
        return ((t ^ (t >> 14)) & _MASK) / 4294967296.0

    def randn(self):
        if self.spare is not None:
            s, self.spare = self.spare, None
            return s
        while True:
            v1 = 2 * self.u() - 1
            v2 = 2 * self.u() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        m = math.sqrt(-2 * math.log(s) / s)
        self.spare = v2 * m
        return v1 * m


@dataclass
class SynthParams:
    Mp: float = 0.35
    Mm: float = -0.55
    noise: float = 0.07            # white pixel noise sigma
    rowNoise: float = 0.10         # AR(1) row-offset sigma
    clutter: int = 4               # number of clutter ridges
    tau: float = 0.06              # per-diamond fractional slope scatter
    Vdead: float = 0.009           # transport dead band (V)
    smear: float = 0.22            # edge-width growth rate per V of bias
    spacing_scatter: float = 0.0   # disorder: fractional per-gap scatter
    spacing_ratio: float = 1.0     # even/odd alternation d2/d1 (spin pairing)
    boost_amp: float = 0.22        # excited-state step amplitude


# grid matching the reference demo
NX, NY = 224, 152
VG = np.linspace(0, 1, NX, endpoint=False) + 0.0            # col i -> vg = i*sx
VG_SX = 1.0 / NX
VSD_MAX, VSD_MIN = 0.12, -0.12
VSD_SY = (VSD_MAX - VSD_MIN) / NY
VSD = VSD_MAX - (np.arange(NY) + 0.5) * VSD_SY               # row centers, descending


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def synth_truth(p: SynthParams, rng: Mulberry32):
    scat, ratio = p.spacing_scatter, p.spacing_ratio
    if scat > 0 or ratio != 1:
        base = 0.21
        d1 = 2 * base / (1 + ratio)
        d2 = ratio * d1
        apex = [0.085]
        k = 0
        while apex[-1] < 0.88 and len(apex) < 8:
            mode = d1 if k % 2 == 0 else d2
            k += 1
            apex.append(apex[-1] + max(0.09, mode * (1 + scat * rng.randn())))
        if apex[-1] > 0.97:
            apex[-1] = 0.97
    else:
        apex = [0.085, 0.28, 0.505, 0.715, 0.925]  # baseline: rng stream untouched
    delta = 0.0025
    diamonds = []
    for i in range(len(apex) - 1):
        mp = p.Mp * (1 + p.tau * rng.randn())
        mm = p.Mm * (1 + p.tau * rng.randn())
        a1, a2 = apex[i], apex[i + 1]
        vgu = (mp * a1 - mm * a2) / (mp - mm)
        vgl = (mm * a1 - mp * a2) / (mm - mp)
        diamonds.append({"a1": a1, "a2": a2, "mp": mp, "mm": mm,
                         "tipUp": {"vg": vgu, "vs": delta + mp * (vgu - a1)},
                         "tipLo": {"vg": vgl, "vs": delta + mm * (vgl - a1)}})
    return {"apex": apex, "diamonds": diamonds, "delta": delta}


def _add_ridges(img, segs):
    W = 1.35
    for s in segs:
        x1 = s["x1"] / VG_SX
        y1 = (VSD_MAX - s["y1"]) / VSD_SY
        x2 = s["x2"] / VG_SX
        y2 = (VSD_MAX - s["y2"]) / VSD_SY
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        xmin = max(0, int(math.floor(min(x1, x2) - 5)))
        xmax = min(NX - 1, int(math.ceil(max(x1, x2) + 5)))
        ymin = max(0, int(math.floor(min(y1, y2) - 5)))
        ymax = min(NY - 1, int(math.ceil(max(y1, y2) + 5)))
        if xmax < xmin or ymax < ymin:
            continue
        jj, ii = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        t = ((ii - x1) * dx + (jj - y1) * dy) / L2 if L2 > 0 else np.zeros_like(ii, float)
        t = np.clip(t, 0, 1)
        qx = x1 + t * dx - ii
        qy = y1 + t * dy - jj
        d2 = qx * qx + qy * qy
        img[ymin:ymax + 1, xmin:xmax + 1] += np.where(
            d2 < 25, s["amp"] * np.exp(-d2 / (2 * W * W)), 0.0)


def _clutter_segments(n, rng):
    segs = []
    for _ in range(n):
        cx, cy = rng.u() * NX, rng.u() * NY
        ang = rng.u() * math.pi
        L = (0.35 + 0.85 * rng.u()) * 42
        amp = 0.18 + 0.14 * rng.u()
        p2x = lambda px: px * VG_SX
        p2y = lambda py: VSD_MAX - py * VSD_SY
        segs.append({"x1": p2x(cx - math.cos(ang) * L / 2), "y1": p2y(cy - math.sin(ang) * L / 2),
                     "x2": p2x(cx + math.cos(ang) * L / 2), "y2": p2y(cy + math.sin(ang) * L / 2),
                     "amp": amp})
    return segs


def synth_current(p: SynthParams, seed: int):
    """Returns (img, vg_axis, vsd_axis, truth). img shape (NY, NX), vsd descending."""
    rng = Mulberry32(seed)
    truth = synth_truth(p, rng)
    w0 = 0.0045
    rho = 0.93
    row_off = np.empty(NY)
    row_off[0] = p.rowNoise * rng.randn()
    for j in range(1, NY):
        row_off[j] = rho * row_off[j - 1] + p.rowNoise * math.sqrt(1 - rho * rho) * rng.randn()

    vg_c = (np.arange(NX) + 0.5) * VG_SX          # pixel centers (synthesis)
    img = np.zeros((NY, NX))
    for j in range(NY):
        v = VSD[j]
        vv = v - truth["delta"]
        av = abs(v)
        dead = 1 / (1 + math.exp(-(av - p.Vdead) / 0.0015))
        imag = max(0.0, min(1.0, (av - p.Vdead) / 0.02)) * (0.55 + 0.45 * av / VSD_MAX)
        wE = w0 + p.smear * abs(vv)
        conduct = np.ones(NX)
        boost = np.zeros(NX)
        for d in truth["diamonds"]:
            if vv >= 0:
                L, R = d["a1"] + vv / d["mp"], d["a2"] + vv / d["mm"]
            else:
                L, R = d["a1"] + vv / d["mm"], d["a2"] + vv / d["mp"]
            if L < R:
                inside = _sig((vg_c - L) / wE) * _sig((R - vg_c) / wE)
                conduct *= (1 - inside)
            if vv > 0.012:
                xE = d["a1"] - 0.045 + vv / d["mp"]
                xM = d["a1"] + vv / d["mp"]
                boost += p.boost_amp * _sig((vg_c - xE) / wE) * _sig((xM - vg_c) / wE)
        img[j] = imag * dead * conduct * (1 + boost)

    clut = np.zeros((NY, NX))
    _add_ridges(clut, _clutter_segments(p.clutter, rng))
    gate = 1 / (1 + np.exp(-(np.abs(VSD) - p.Vdead) / 0.0015))
    img += gate[:, None] * clut

    # pixel noise drawn in the same (row-major) order as the reference
    wn = np.array([rng.randn() for _ in range(NY * NX)]).reshape(NY, NX)
    img += row_off[:, None] + p.noise * wn

    vg_axis = np.arange(NX) * VG_SX               # analysis convention: vg[i] = i*sx
    return img, vg_axis, VSD.copy(), truth


def true_ec_per_diamond(truth):
    return [{"vgL": d["a1"], "vgR": d["a2"],
             "Ec": (abs(d["tipUp"]["vs"] - truth["delta"])
                    + abs(d["tipLo"]["vs"] - truth["delta"])) / 2}
            for d in truth["diamonds"]]
