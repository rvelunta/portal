"""Command line: qua script in, scope diagram out.

    python3 -m qua_scope script.py                 # -> script.scope.svg
    python3 -m qua_scope script.py --text          # ASCII scope in the terminal
    python3 -m qua_scope script.py --dump          # the traced program tree
    python3 -m qua_scope script.py -i 4 --modulate # more loop iterations, IF shown
"""

import argparse
import json
import os
import sys

from .interp import Interpreter, RunConfig
from .loader import load_script
from .machine import Machine
from .render_svg import render_svg
from .render_text import render_text
from .textdump import program_to_text
from .timeline import fmt_time


def parse_time(s):
    """'500', '500ns', '2us', '1.5ms', '1s' -> nanoseconds."""
    if s is None:
        return 0.0
    s = str(s).strip().lower().replace("µ", "u")
    for suffix, mult in (("ns", 1), ("us", 1e3), ("ms", 1e6), ("s", 1e9)):
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * mult
    return float(s)


def build_parser():
    p = argparse.ArgumentParser(
        prog="qua-scope",
        description="Interpret a QUA (Quantum Machines) script and draw the "
                    "pulse sequence it would play.")
    p.add_argument("script", help="path to the .py QUA script")
    p.add_argument("-o", "--out", help="output .svg (default: <script>.scope.svg)")
    p.add_argument("--text", action="store_true", help="ASCII scope on stdout")
    p.add_argument("--dump", action="store_true",
                   help="print the traced QUA program instead of drawing")
    p.add_argument("--json", dest="json_out", metavar="FILE",
                   help="also write the timeline as JSON")
    p.add_argument("-i", "--iterations", type=int, default=2,
                   help="loop iterations to unroll per loop (default 2)")
    p.add_argument("--full", action="store_true",
                   help="unroll loops completely (bounded by --max-segments)")
    p.add_argument("--tmax", default=None,
                   help="stop after this much sequence time, e.g. 2us")
    p.add_argument("--tmin", default="0", help="start the view at this time")
    p.add_argument("--max-segments", type=int, default=20000,
                   help="pulse budget before unrolling gives up")
    p.add_argument("--elements", help="comma-separated elements to show")
    p.add_argument("--no-compress", dest="compress", action="store_false",
                   help="keep the time axis strictly linear instead of "
                        "collapsing long idle stretches")
    p.add_argument("--modulate", action="store_true",
                   help="draw the IF-modulated output instead of the envelope")
    p.add_argument("--branch", choices=("auto", "then", "else"), default="auto",
                   help="which side of a run-time if_ to draw (default auto)")
    p.add_argument("--program", type=int, default=-1,
                   help="index of the program() block to draw (default: last)")
    p.add_argument("--config", help="where to find the QM config, as module:name")
    p.add_argument("--width", type=int, default=1180, help="SVG plot width (px)")
    p.add_argument("--lane-height", type=int, default=62)
    p.add_argument("--theme", choices=("auto", "light", "dark"), default="auto")
    p.add_argument("--title", help="diagram title (default: script name)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="wall-clock bound on running the script")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def resolve_config(spec):
    """--config module:name -- import the module and pull the dict out."""
    mod, _, name = spec.partition(":")
    sys.path.insert(0, os.getcwd())
    m = __import__(mod, fromlist=["*"])
    cfg = getattr(m, name or "config")
    if not isinstance(cfg, dict):
        raise SystemExit(f"--config {spec}: not a dict")
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    log = (lambda *a: None) if args.quiet else (
        lambda *a: print(*a, file=sys.stderr))

    extra = resolve_config(args.config) if args.config else None
    res = load_script(args.script, timeout=args.timeout, extra_config=extra)
    for w in res.warnings:
        log(f"note: {w}")
    if not res.programs:
        raise SystemExit(
            f"{args.script}: no `with program()` block was traced. If the "
            f"program is built inside a function, call that function at "
            f"module level (or under `if __name__ == '__main__':`).")
    if len(res.programs) > 1:
        log(f"note: {len(res.programs)} program blocks traced; drawing "
            f"#{args.program % len(res.programs) + 1}")
    prog = res.programs[args.program]

    if args.dump:
        print(program_to_text(prog))
        return 0

    if not res.configs:
        raise SystemExit(
            f"{args.script}: found the program but no QM configuration dict "
            f"(needs 'elements' and 'pulses'). Pass --config module:name.")

    machine = Machine(res.config)
    cfg = RunConfig(
        max_iterations=10 ** 9 if args.full else max(1, args.iterations),
        max_segments=args.max_segments,
        max_time_ns=parse_time(args.tmax),
        branch=args.branch,
    )
    tl = Interpreter(machine, cfg).run(prog)

    if args.elements:
        keep = [e.strip() for e in args.elements.split(",")]
        missing = [e for e in keep if e not in tl.elements]
        if missing:
            log(f"note: not used by this program: {', '.join(missing)}")
        tl.elements = [e for e in tl.elements if e in keep]

    for w in tl.warnings:
        log(f"note: {w}")
    log(f"scope: {tl.summary()}")

    t0 = parse_time(args.tmin)
    t1 = parse_time(args.tmax) or tl.duration

    if args.text:
        print(render_text(tl, width=_term_width(), t0=t0, t1=t1,
                          modulate=args.modulate, compress=args.compress))

    out = args.out
    if out is None and not args.text:
        out = os.path.splitext(os.path.basename(args.script))[0] + ".scope.svg"
    if out:
        svg = render_svg(
            tl, width=args.width, lane_height=args.lane_height, t0=t0, t1=t1,
            modulate=args.modulate, theme=args.theme, compress=args.compress,
            title=args.title or os.path.basename(args.script),
            subtitle=f"{tl.summary()}  |  "
                     f"{'envelope' if not args.modulate else 'IF-modulated'}"
                     f", {'all' if args.full else args.iterations} "
                     f"iteration(s) per loop")
        with open(out, "w") as f:
            f.write(svg)
        log(f"wrote {out}  ({fmt_time(t1 - t0)} of sequence)")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(timeline_to_dict(tl), f, indent=1)
        log(f"wrote {args.json_out}")
    return 0


def timeline_to_dict(tl):
    return {
        "duration_ns": tl.duration,
        "elements": tl.elements,
        "segments": [
            {"element": s.element, "kind": s.kind, "t0": s.t0, "t1": s.t1,
             "op": s.op, "pulse": s.pulse, "amp": s.amp, "freq": s.freq,
             "phase": s.phase, "channels": list(s.waves), "loc": s.loc,
             "iteration": list(s.iteration), "runtime": s.runtime}
            for s in tl.segments],
        "acquisitions": [
            {"element": a.element, "t0": a.t0, "t1": a.t1, "label": a.label}
            for a in tl.acquisitions],
        "markers": [
            {"kind": m.kind, "t": m.t, "elements": m.elements, "label": m.label}
            for m in tl.markers],
        "breaks": [{"t": b.t, "skipped": b.skipped, "label": b.label}
                   for b in tl.breaks],
        "levels": [{"element": L.element, "channel": L.channel, "t": L.t,
                    "level": L.level, "reason": L.reason} for L in tl.levels],
        "warnings": tl.warnings,
    }


def _term_width():
    try:
        return max(60, min(160, os.get_terminal_size().columns))
    except OSError:
        return 100


if __name__ == "__main__":
    raise SystemExit(main())
