"""ASCII scope -- the terminal view of a Timeline.

Same data and the same (optionally gap-collapsed) axis as the SVG, sampled into
character columns: one lane per element channel, block glyphs sized against that
lane's own peak, plus rows for ADC windows and markers. Good enough to check
timing at a glance over ssh, and it is what `--text` prints.
"""

from .axis import TimeAxis
from .timeline import fmt_freq, fmt_time

BLOCKS = " ▁▂▃▄▅▆▇█"


def render_text(tl, width=100, t0=0.0, t1=None, modulate=False, label_w=18,
                compress=True):
    t1 = tl.duration if t1 is None else t1
    if t1 <= t0:
        t1 = t0 + 1
    cols = max(20, width - label_w - 1)
    ax = TimeAxis(tl, t0, t1, cols, x0=0.0, compress=compress, gap_px=3,
                  min_gap_ns=40)
    spans = [(ax.t_of(i), max(ax.t_of(i + 1), ax.t_of(i) + 1e-9))
             for i in range(cols)]
    gap_cols = set()
    for xc, _ga, _gb in ax.breaks:
        for i in range(int(xc - ax.gap_px / 2), int(xc + ax.gap_px / 2) + 1):
            if 0 <= i < cols:
                gap_cols.add(i)
    lines = []

    def lane(label, cells):
        lines.append(f"{label[:label_w]:<{label_w}}|{''.join(cells)}")

    for el in tl.elements:
        e = tl.machine.element(el) if tl.machine else None
        head = el
        if e is not None and e.intermediate_frequency:
            head += f" @{fmt_freq(e.intermediate_frequency)}"
        lines.append(head)
        for ch in (tl.channels(el) or ["single"]):
            raw = tl.peak(el, ch)
            peak = raw or 1.0
            cells = []
            for i, (ta, tb) in enumerate(spans):
                if i in gap_cols:
                    cells.append("~")
                    continue
                lo, hi = tl.range_over(el, ch, ta, tb, modulate)
                v = max(abs(lo), abs(hi)) / peak
                cells.append(BLOCKS[min(8, int(round(v * 8)))])
            lane(f"  {ch} {f'+-{raw:.3g}V' if raw else 'flat'}", cells)
        acq = [a for a in tl.acquisitions if a.element == el]
        if acq:
            cells = [" "] * cols
            for a in acq:
                for i in range(_col(ax, a.t0, cols), _col(ax, a.t1, cols)):
                    cells[i] = "="
            lane("  acquire", cells)

    cells = [" "] * cols
    for m in tl.markers:
        if m.kind == "wait":
            continue
        i = _col(ax, m.t, cols)
        if 0 <= i < cols:
            cells[i] = {"align": "A", "pause": "P", "freq": "f", "frame": "p",
                        "branch": "?", "trigger": "T", "dc": "d"}.get(m.kind, "*")
    for b in tl.breaks:
        i = _col(ax, b.t, cols)
        if 0 <= i < cols:
            cells[i] = "/"
    lane("markers", cells)

    ticks = [" "] * cols
    axis = [" "] * cols
    for _t, x, lbl in ax.ticks(target=6, min_spacing=13):
        i = int(x)
        if not (0 <= i < cols):
            continue
        ticks[i] = "|"
        for j, c in enumerate(lbl):
            if i + j < cols:
                axis[i + j] = c
    lane("time", ticks)
    lane("", axis)

    legend = []
    if any(m.kind == "align" for m in tl.markers):
        legend.append("A align")
    if tl.breaks:
        legend.append("/ loop truncated")
    if any(m.kind == "branch" for m in tl.markers):
        legend.append("? run-time branch")
    if ax.breaks:
        collapsed = sum(gb - ga for _x, ga, gb in ax.breaks)
        legend.append(f"~ idle collapsed ({fmt_time(collapsed)} total)")
    if legend:
        lines.append("  ".join(legend))
    return "\n".join(lines)


def _col(ax, t, cols):
    return max(0, min(cols - 1, int(ax.x_of(t))))
