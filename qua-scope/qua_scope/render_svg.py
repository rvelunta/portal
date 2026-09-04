"""Timeline -> SVG scope diagram.

One lane per analog channel (a mixed-input element gets I and Q), drawn as a
filled envelope against that lane's own peak so a 10 mV readout and a 250 mV pi
pulse are both readable. Overlaid on top: measurement windows (shifted by
time_of_flight, widened by smearing), align markers as vertical guides, and
hatched bands where loop unrolling was truncated.

Every segment carries an SVG <title>, so hovering a pulse in a browser tells you
the operation, its amplitude scaling, the IF, the loop iteration it belongs to,
and the source line that played it.

Zero dependencies: this writes the XML by hand.
"""

import math
from xml.sax.saxutils import escape

from .axis import TimeAxis
from .timeline import fmt_freq, fmt_time

PALETTE = ["#4c6ef5", "#12b886", "#f76707", "#ae3ec9", "#e8590c",
           "#1098ad", "#f59f00", "#e64980", "#2f9e44", "#7048e8"]

LABEL_W = 176
PAD_X = 14
HEAD_H = 54
MARKER_H = 18
AXIS_H = 30


def render_svg(tl, width=1180, lane_height=62, t0=0.0, t1=None, modulate=False,
               theme="auto", title="qua scope", subtitle="", compress=True):
    t1 = tl.duration if t1 is None else t1
    if t1 <= t0:
        t1 = t0 + 1.0
    plot_w = max(320, width - LABEL_W - 2 * PAD_X)
    ax = TimeAxis(tl, t0, t1, plot_w, x0=LABEL_W + PAD_X, compress=compress)

    lanes = []          # (element, channel, color)
    for i, el in enumerate(tl.elements):
        color = PALETTE[i % len(PALETTE)]
        for ch in (tl.channels(el) or ["single"]):
            lanes.append((el, ch, color))
    if not lanes:
        lanes = [("(no output)", "single", PALETTE[0])]

    plot_h = len(lanes) * lane_height
    footer = _footer_lines(tl, modulate, len(ax.breaks))
    height = (HEAD_H + MARKER_H + plot_h + AXIS_H + 14 * len(footer) + 16)

    x_of = ax.x_of
    o = []
    add = o.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height:.0f}" viewBox="0 0 {width} {height:.0f}" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">')
    add(_style(theme))
    add(_defs())
    add(f'<rect class="bg" x="0" y="0" width="{width}" height="{height:.0f}"/>')

    # ---- header
    add(f'<text class="title" x="{PAD_X}" y="24">{escape(title)}</text>')
    if subtitle:
        add(f'<text class="sub" x="{PAD_X}" y="40">{escape(subtitle)}</text>')

    top = HEAD_H + MARKER_H
    # ---- time grid
    for t, x, label in ax.ticks():
        add(f'<line class="grid" x1="{x:.1f}" y1="{top:.0f}" x2="{x:.1f}" '
            f'y2="{top + plot_h:.0f}"/>')
        add(f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 15:.0f}" '
            f'text-anchor="middle">{escape(label)}</text>')
    add(f'<line class="axis" x1="{LABEL_W + PAD_X}" y1="{top + plot_h:.0f}" '
        f'x2="{LABEL_W + PAD_X + plot_w}" y2="{top + plot_h:.0f}"/>')

    # ---- lanes
    for k, (el, ch, color) in enumerate(lanes):
        y = top + k * lane_height
        add(_lane(tl, el, ch, color, y, lane_height, ax, t0, t1, plot_w,
                  modulate, k, lanes))

    # ---- collapsed idle gaps
    for xc, ga, gb in ax.breaks:
        add(f'<rect class="gap" x="{xc - ax.gap_px / 2:.1f}" y="{top:.0f}" '
            f'width="{ax.gap_px}" height="{plot_h:.0f}">'
            f'<title>idle {fmt_time(gb - ga)} collapsed\n'
            f'{fmt_time(ga)} -> {fmt_time(gb)}</title></rect>')
        add(f'<text class="gaplbl" x="{xc:.1f}" y="{top + plot_h + 15:.0f}" '
            f'text-anchor="middle">{escape(fmt_time(gb - ga))}</text>')

    # ---- measurement windows on top of their element's lanes
    for a in tl.acquisitions:
        rows = [k for k, (e, _c, _col) in enumerate(lanes) if e == a.element]
        if not rows or a.t1 < t0 or a.t0 > t1:
            continue
        ya = top + rows[0] * lane_height + 2
        yb = top + (rows[-1] + 1) * lane_height - 2
        xa, xb = max(x_of(a.t0), LABEL_W + PAD_X), min(x_of(a.t1),
                                                       LABEL_W + PAD_X + plot_w)
        add(f'<rect class="acq" x="{xa:.1f}" y="{ya:.0f}" '
            f'width="{max(1.0, xb - xa):.1f}" height="{yb - ya:.0f}">'
            f'<title>{escape(a.label)}\nADC window {fmt_time(a.t0)} - '
            f'{fmt_time(a.t1)} ({fmt_time(a.t1 - a.t0)})\n{escape(a.loc)}</title>'
            f'</rect>')
        if xb - xa > 46:
            add(f'<text class="acqlbl" x="{xa + 4:.1f}" y="{ya + 11:.0f}">'
                f'acquire</text>')

    # ---- markers (align, pause, run-time branch, frequency/frame updates)
    add(_markers(tl, x_of, t0, t1, top, plot_h, lanes, lane_height))

    # ---- loop-truncation bands
    for b in tl.breaks:
        if b.t < t0 or b.t > t1:
            continue
        x = x_of(b.t)
        add(f'<rect class="brk" x="{x - 5:.1f}" y="{top:.0f}" width="10" '
            f'height="{plot_h:.0f}"><title>{escape(b.label)}</title></rect>')
        n = "?" if b.skipped is None else f"{b.skipped}"
        add(f'<text class="brklbl" x="{x + 8:.1f}" y="{top - 5:.0f}">'
            f'&#215;{escape(n)} more</text>')

    # ---- footer
    fy = top + plot_h + AXIS_H
    for i, line in enumerate(footer):
        add(f'<text class="note" x="{PAD_X}" y="{fy + 14 * i:.0f}">'
            f'{escape(line)}</text>')
    add("</svg>")
    return "\n".join(o)


# --------------------------------------------------------------------- pieces

def _lane(tl, el, ch, color, y, h, ax, t0, t1, plot_w, modulate, k, lanes):
    o = []
    x_of = ax.x_of
    cy = y + h / 2
    half = h / 2 - 9
    raw_peak = tl.peak(el, ch)
    peak = raw_peak or 1.0                # flat channel: keep the lane, no scale
    first_of_element = k == 0 or lanes[k - 1][0] != el

    o.append(f'<rect class="lane{" alt" if k % 2 else ""}" x="{LABEL_W + PAD_X}"'
             f' y="{y:.0f}" width="{plot_w}" height="{h}"/>')
    o.append(f'<line class="zero" x1="{LABEL_W + PAD_X}" y1="{cy:.1f}" '
             f'x2="{LABEL_W + PAD_X + plot_w}" y2="{cy:.1f}"/>')

    # label gutter
    el_obj = tl.machine.element(el) if tl.machine else None
    if first_of_element:
        o.append(f'<rect x="{PAD_X}" y="{y + 4:.0f}" width="3" '
                 f'height="{h - 8:.0f}" fill="{color}" rx="1.5"/>')
        o.append(f'<text class="ellbl" x="{PAD_X + 10}" y="{y + 16:.0f}">'
                 f'{escape(_clip(el, 15))}<title>{escape(el)}</title></text>')
        meta = []
        if el_obj is not None and el_obj.intermediate_frequency:
            meta.append("IF " + fmt_freq(el_obj.intermediate_frequency))
        if el_obj is not None and el_obj.port_label:
            meta.append(el_obj.port_label)
        if meta:
            full = "  ".join(meta)
            o.append(f'<text class="elmeta" x="{PAD_X + 10}" y="{y + 29:.0f}">'
                     f'{escape(_clip(full, 17))}'
                     f'<title>{escape(full)}</title></text>')
    chlbl = ch if ch != "single" else "out"
    scale = f"&#177;{raw_peak:.3g}V" if raw_peak else "flat 0V"
    o.append(f'<text class="chlbl" x="{LABEL_W + PAD_X - 6}" y="{cy + 3:.0f}" '
             f'text-anchor="end">{escape(chlbl)} {scale}</text>')

    # waveform: min/max per pixel column, drawn as a closed band. Columns are
    # mapped back through the axis, so a collapsed gap costs a few columns and
    # a dense burst still gets one column per pixel.
    cols = int(plot_w)
    pts = []
    for i in range(cols + 1):
        x = ax.x0 + i
        ta, tb = ax.t_of(x), ax.t_of(x + 1)
        lo, hi = tl.range_over(el, ch, ta, max(tb, ta + 1e-9), modulate)
        pts.append((x, cy - hi / peak * half, cy - lo / peak * half))
    pts = _simplify(pts)
    ups = " ".join(f"{x:.0f},{u:.2f}" for x, u, _d in pts)
    downs = " ".join(f"{x:.0f},{d:.2f}" for x, _u, d in reversed(pts))
    o.append(f'<polygon class="wave" fill="{color}" stroke="{color}" '
             f'points="{ups} {downs}"/>')

    # per-segment hit areas + labels
    mine = [s for s in tl.segments
            if s.element == el and ch in s.waves and s.t1 >= t0 and s.t0 <= t1]
    for i, s in enumerate(mine):
        xa, xb = x_of(s.t0), x_of(s.t1)
        w = max(1.0, xb - xa)
        o.append(f'<rect class="seg{" rt" if s.runtime else ""}" x="{xa:.1f}" '
                 f'y="{y + 2:.0f}" width="{w:.1f}" height="{h - 4:.0f}">'
                 f'<title>{escape(_seg_title(s, el_obj))}</title></rect>')
        if not first_of_element:
            continue
        label = s.op + (s.amp_label or "")
        need = 6.2 * len(label)
        room = (x_of(mine[i + 1].t0) if i + 1 < len(mine)
                else LABEL_W + PAD_X + plot_w) - xa
        if w >= need:                       # fits inside the pulse: centre it
            o.append(f'<text class="seglbl" x="{(xa + xb) / 2:.1f}" '
                     f'y="{y + 12:.0f}" text-anchor="middle">'
                     f'{escape(label)}</text>')
        elif room >= need + 4:              # narrow pulse: label the idle space
            o.append(f'<text class="seglbl thin" x="{xa + 2:.1f}" '
                     f'y="{y + 12:.0f}">{escape(label)}</text>')
            o.append(f'<line class="stub" x1="{xa:.1f}" y1="{y + 14:.0f}" '
                     f'x2="{xa:.1f}" y2="{y + h - 4:.0f}"/>')
    return "\n".join(o)


def _simplify(pts, tol=0.15):
    """Drop columns that repeat the previous value -- a flat lane is two points,
    not twelve hundred. Edges are always kept, so pulse boundaries stay square."""
    out = [pts[0]]
    for i, p in enumerate(pts[1:-1], 1):
        prev, nxt = out[-1], pts[i + 1]
        flat = abs(p[1] - prev[1]) < tol and abs(p[2] - prev[2]) < tol
        edge = abs(nxt[1] - p[1]) > tol or abs(nxt[2] - p[2]) > tol
        if not flat or edge:
            out.append(p)
    out.append(pts[-1])
    return out


def _clip(s, n):
    return s if len(s) <= n else s[:n - 1] + "…"


def _seg_title(s, el_obj):
    lines = [f"{s.op}  ({s.pulse})",
             f"{fmt_time(s.t0)} -> {fmt_time(s.t1)}   {fmt_time(s.duration)}",
             f"amp x{s.amp:.4g}"]
    if s.freq:
        lines.append(f"IF {fmt_freq(s.freq)}")
    if s.phase:
        lines.append(f"frame {s.phase / math.pi:.3g}pi")
    if s.iteration:
        lines.append("iteration " + ".".join(map(str, s.iteration)))
    if s.runtime:
        lines.append("inside a run-time-dependent branch")
    if s.loc:
        lines.append(s.loc)
    return "\n".join(lines)


def _markers(tl, x_of, t0, t1, top, plot_h, lanes, lane_height):
    o = []
    glyph = {"align": "A", "pause": "P", "freq": "f", "frame": "φ",
             "branch": "?", "trigger": "T", "dc": "d", "phase": "φ"}
    lastx = {}
    for m in tl.markers:
        if m.t < t0 or m.t > t1 or m.kind == "wait":
            continue
        x = x_of(m.t)
        cls = "mk " + m.kind
        rows = [k for k, (e, _c, _col) in enumerate(lanes)
                if not m.elements or e in m.elements]
        if m.kind in ("align", "pause", "branch") and rows:
            ya = top + rows[0] * lane_height
            yb = top + (rows[-1] + 1) * lane_height
            o.append(f'<line class="{cls}" x1="{x:.1f}" y1="{ya:.0f}" '
                     f'x2="{x:.1f}" y2="{yb:.0f}"/>')
        # keep the glyph row readable when markers pile up
        key = round(x / 9)
        if lastx.get(key) == m.kind:
            continue
        lastx[key] = m.kind
        o.append(f'<text class="mklbl {m.kind}" x="{x:.1f}" '
                 f'y="{top - 6:.0f}" text-anchor="middle">'
                 f'{escape(glyph.get(m.kind, "*"))}'
                 f'<title>{escape(m.label)}\n{escape(", ".join(m.elements))}'
                 f'\n{fmt_time(m.t)}  {escape(m.loc)}</title></text>')
    return "\n".join(o)


def _footer_lines(tl, modulate, gaps=0):
    out = []
    legend = ["A align", "P pause", "? run-time branch", "f IF update",
              "φ frame", "shaded = ADC window", "hatched = loop truncated"]
    if gaps:
        legend.append("zigzag = idle time collapsed (axis is not linear there)")
    out.append("  ".join(legend))
    if not modulate:
        out.append("Traces are pulse envelopes (--modulate draws the IF-mixed "
                   "output). Timing is the ideal QUA model: no compiler "
                   "overhead, no analog filter delay.")
    for w in tl.warnings[:6]:
        out.append("note: " + w)
    return out


def _style(theme):
    light = """
      .bg{fill:#fbfbfd}.title{fill:#151823;font-size:15px;font-weight:700}
      .sub{fill:#6b7280;font-size:10.5px}.note{fill:#8a90a0;font-size:10px}
      .lane{fill:#ffffff;stroke:#e7e9f0}.lane.alt{fill:#f7f8fc;stroke:#e7e9f0}
      .zero{stroke:#c9cede;stroke-dasharray:2 3}.grid{stroke:#e7e9f0}
      .axis{stroke:#aeb4c6}.tick{fill:#8a90a0;font-size:9.5px}
      .ellbl{fill:#151823;font-size:11.5px;font-weight:700}
      .elmeta{fill:#8a90a0;font-size:9px}.chlbl{fill:#6b7280;font-size:9.5px}
      .seglbl{fill:#151823;font-size:9.5px}.seglbl.thin{fill:#5a6070;font-size:9px}
      .stub{stroke:#aeb4c6;stroke-width:.8;stroke-dasharray:1 2}
      .acq{fill:url(#hatch);stroke:#12b886;stroke-dasharray:3 2}
      .acqlbl{fill:#0b7a5c;font-size:9px}
      .mk.align{stroke:#e64980}.mk.pause{stroke:#f76707}.mk.branch{stroke:#ae3ec9}
      .mklbl{font-size:10px;fill:#6b7280}.mklbl.align{fill:#e64980}
      .mklbl.branch{fill:#ae3ec9}.mklbl.pause{fill:#f76707}
      .brk{fill:url(#zig);stroke:none}.brklbl{fill:#8a90a0;font-size:9px}
      .gap{fill:url(#gapzig);stroke:#c9cede;stroke-dasharray:2 2}
      .gaplbl{fill:#8a90a0;font-size:8.5px}
    """
    dark = """
      .bg{fill:#0f1117}.title{fill:#f2f4f8}.sub{fill:#9aa1b2}.note{fill:#767d8f}
      .lane{fill:#161923;stroke:#242938}.lane.alt{fill:#12151e;stroke:#242938}
      .zero{stroke:#39405a}.grid{stroke:#232838}.axis{stroke:#4a5268}
      .tick{fill:#8a92a6}.ellbl{fill:#f2f4f8}.elmeta{fill:#767d8f}
      .chlbl{fill:#9aa1b2}.seglbl{fill:#e7eaf2}.acqlbl{fill:#3ddbaa}
      .mklbl{fill:#9aa1b2}
    """
    common = """
      .wave{fill-opacity:.32;stroke-width:1.1;stroke-linejoin:round}
      .seg{fill:transparent;stroke:none;pointer-events:all}
      .seg.rt{fill:#ae3ec9;fill-opacity:.07}
      .acq{fill-opacity:.5;stroke-width:1}
      .mk{stroke-width:1;stroke-dasharray:3 3;opacity:.75}
      text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
    """
    if theme == "light":
        css = light + common
    elif theme == "dark":
        css = light + common + dark
    else:
        css = light + common + "@media (prefers-color-scheme: dark){" + dark + "}"
    return f"<style>{css}</style>"


def _defs():
    return (
        '<defs>'
        '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)">'
        '<rect width="6" height="6" fill="#12b886" fill-opacity="0.10"/>'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="#12b886" stroke-width="1.6" '
        'stroke-opacity="0.35"/></pattern>'
        '<pattern id="zig" width="6" height="6" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(20)">'
        '<rect width="6" height="6" fill="#8a90a0" fill-opacity="0.10"/>'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="#8a90a0" stroke-width="2" '
        'stroke-opacity="0.45"/></pattern>'
        '<pattern id="gapzig" width="7" height="7" patternUnits="userSpaceOnUse">'
        '<path d="M0,7 L3.5,0 L7,7" fill="none" stroke="#aeb4c6" '
        'stroke-width="1" stroke-opacity="0.55"/></pattern>'
        '</defs>')
