"""The time axis, with idle gaps optionally collapsed.

A QUA sequence spends most of its wall time waiting: a 40 ns pi pulse followed
by a 10 us thermal-decay wait. Drawn on a linear axis the pulse is a quarter of
a pixel wide and the picture says nothing. So by default long idle stretches --
where no element is playing or acquiring -- are collapsed to a fixed narrow
band, marked with a break, exactly like the gap-skip on a logic analyser.

`x_of`/`t_of` are the piecewise-linear map (and its inverse) between sequence
time and pixels; every renderer goes through them, so nothing has to know
whether compression is on.
"""

from .timeline import fmt_time

GAP_PX = 26                # width given to one collapsed gap
MIN_GAP_NS = 40            # never collapse anything shorter than this


class TimeAxis:
    def __init__(self, tl, t0, t1, plot_w, x0=0.0, compress=True,
                 gap_px=GAP_PX, min_gap_ns=MIN_GAP_NS):
        self.t0, self.t1, self.plot_w, self.x0 = t0, t1, plot_w, x0
        self.gap_px = gap_px
        self.nodes = []            # (kind, ta, tb, xa, xb), kind: 'span'|'gap'
        busy = _busy_intervals(tl, t0, t1)
        gaps = _complement(busy, t0, t1)
        self.compressed = []
        if compress and gaps:
            self.compressed = _choose_gaps(gaps, t0, t1, plot_w, gap_px,
                                           min_gap_ns)
            # Hundreds of collapsed gaps (a fully unrolled averaging loop) would
            # otherwise claim more width than the plot has: shrink the bands.
            if self.compressed and gap_px * len(self.compressed) > 0.4 * plot_w:
                self.gap_px = max(2.0, 0.4 * plot_w / len(self.compressed))
        self._build()

    # ------------------------------------------------------------ build/map
    def _build(self):
        kept = self.t1 - self.t0 - sum(b - a for a, b in self.compressed)
        width = self.plot_w - self.gap_px * len(self.compressed)
        scale = (width / kept) if kept > 0 else 0.0
        self.scale = scale
        x, t = self.x0, self.t0
        for ga, gb in self.compressed + [(self.t1, self.t1)]:
            if ga > t:
                w = (ga - t) * scale
                self.nodes.append(("span", t, ga, x, x + w))
                x += w
            if gb > ga:
                self.nodes.append(("gap", ga, gb, x, x + self.gap_px))
                x += self.gap_px
            t = gb
        if not self.nodes:
            self.nodes.append(("span", self.t0, self.t1, self.x0,
                               self.x0 + self.plot_w))

    def x_of(self, t):
        if t <= self.t0:
            return self.x0
        for kind, ta, tb, xa, xb in self.nodes:
            if t <= tb:
                if tb == ta:
                    return xa
                return xa + (t - ta) / (tb - ta) * (xb - xa)
        return self.nodes[-1][4]

    def t_of(self, x):
        for kind, ta, tb, xa, xb in self.nodes:
            if x <= xb:
                if xb == xa:
                    return ta
                return ta + (x - xa) / (xb - xa) * (tb - ta)
        return self.t1

    @property
    def breaks(self):
        """(x_center, t_start, t_end) for each collapsed gap."""
        return [((xa + xb) / 2, ta, tb)
                for kind, ta, tb, xa, xb in self.nodes if kind == "gap"]

    def ticks(self, target=9, min_spacing=46):
        """Nice time ticks that respect the breaks: each visible span gets at
        least its own start label, so a collapsed gap never hides where the
        clock jumped to."""
        spans = [n for n in self.nodes if n[0] == "span"]
        if not spans:
            return []
        total = sum(tb - ta for _k, ta, tb, _xa, _xb in spans)
        step = _nice_step(total / max(1, target))
        out, last_x = [], -1e9
        for _k, ta, tb, xa, xb in spans:
            local = []
            t = (int(ta / step) + (0 if abs(ta % step) < 1e-9 else 1)) * step
            while t <= tb + 1e-9:
                local.append((t, xa + (t - ta) / (tb - ta) * (xb - xa)
                              if tb > ta else xa))
                t += step
            if not local:
                local = [(ta, xa)]
            for tv, xv in local:
                if xv - last_x >= min_spacing:
                    out.append((tv, xv, fmt_time(tv)))
                    last_x = xv
        return out


# ------------------------------------------------------------------ helpers

def _busy_intervals(tl, t0, t1):
    """When is anything actually on an output or in an ADC window?"""
    iv = []
    for s in tl.segments:
        iv.append((s.t0, s.t1))
    for a in tl.acquisitions:
        iv.append((a.t0, a.t1))
    for m in tl.markers:                    # keep markers visible
        if m.kind in ("align", "pause", "branch", "freq", "frame", "dc"):
            iv.append((m.t - 1, m.t + 1))
    for L in tl.levels:
        iv.append((L.t - 1, L.t + 1))
    iv = [(max(a, t0), min(b, t1)) for a, b in iv if b >= t0 and a <= t1]
    iv.sort()
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _complement(busy, t0, t1):
    gaps, t = [], t0
    for a, b in busy:
        if a > t:
            gaps.append((t, a))
        t = max(t, b)
    if t < t1:
        gaps.append((t, t1))
    return gaps


def _choose_gaps(gaps, t0, t1, plot_w, gap_px, min_gap_ns):
    """Collapse the gaps that would otherwise eat the picture: iterate, because
    collapsing one gap rescales everything else."""
    chosen = []
    for _ in range(6):
        kept = (t1 - t0) - sum(b - a for a, b in chosen)
        width = plot_w - gap_px * len(chosen)
        if kept <= 0 or width <= 0:
            break
        scale = width / kept
        new = [(a, b) for a, b in gaps
               if (a, b) not in chosen
               and (b - a) >= min_gap_ns and (b - a) * scale > gap_px * 1.5]
        if not new:
            break
        chosen = sorted(chosen + new)
    return chosen


def _nice_step(raw):
    import math
    if raw <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag
