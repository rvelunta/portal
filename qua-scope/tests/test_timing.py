"""The contract: QUA timing semantics, unrolling, and the loader.

Run: python3 tests/test_timing.py   (from the qua-scope directory)

These are the claims the diagram rests on -- per-element cursors, align as a
max, wait in clock cycles, amp()/duration= applied, measurement windows shifted
by time_of_flight and widened by smearing, sticky levels, loop truncation
counts, and run-time-dependent branches marked rather than guessed at.
"""

import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import qua_scope.qua as q                                    # noqa: E402
from qua_scope import (Interpreter, Machine, RunConfig,      # noqa: E402
                       load_script, render_svg, render_text)

failures = []


def check(name, cond, msg=""):
    print(f"    [{'ok ' if cond else 'FAIL'}] {name} {msg}")
    if not cond:
        failures.append(f"{name} {msg}")


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


CONFIG = {
    "version": 1,
    "controllers": {"con1": {"analog_outputs": {i: {"offset": 0.0}
                                                for i in range(1, 6)}}},
    "elements": {
        "qubit": {
            "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2),
                          "lo_frequency": 5e9},
            "intermediate_frequency": 100e6,
            "operations": {"pi": "pi_pulse", "long": "long_pulse"},
        },
        "res": {
            "mixInputs": {"I": ("con1", 3), "Q": ("con1", 4),
                          "lo_frequency": 7e9},
            "intermediate_frequency": 50e6,
            "operations": {"readout": "readout_pulse"},
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 24,
            "smearing": 8,
        },
        "gate": {
            "singleInput": {"port": ("con1", 5)},
            "operations": {"step": "step_pulse"},
            "hold_offset": {"duration": 1},
        },
    },
    "pulses": {
        "pi_pulse": {"operation": "control", "length": 40,
                     "waveforms": {"I": "const_wf", "Q": "zero_wf"}},
        "long_pulse": {"operation": "control", "length": 1000,
                       "waveforms": {"I": "const_wf", "Q": "zero_wf"}},
        "readout_pulse": {"operation": "measurement", "length": 400,
                          "waveforms": {"I": "ro_wf", "Q": "zero_wf"}},
        "step_pulse": {"operation": "control", "length": 16,
                       "waveforms": {"single": "step_wf"}},
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "ro_wf": {"type": "constant", "sample": 0.05},
        "step_wf": {"type": "constant", "sample": 0.02},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
}


def run(program, iterations=2, branch="auto"):
    return Interpreter(Machine(CONFIG),
                       RunConfig(max_iterations=iterations,
                                 branch=branch)).run(program)


def segs(tl, element=None, kind=None):
    return [s for s in tl.segments
            if (element is None or s.element == element)
            and (kind is None or s.kind == kind)]


# ---------------------------------------------------------------- suites

def test_cursors():
    print("  [per-element cursors]")
    with q.program() as p:
        q.play("pi", "qubit")            # qubit 0 -> 40
        q.play("pi", "qubit")            # qubit 40 -> 80
        q.play("readout", "res")         # res starts at 0, independent
        q.wait(25, "qubit")              # +100 ns
        q.align("qubit", "res")          # both -> max(180, 400) = 400
        q.play("pi", "qubit")
    tl = run(p)
    qs = segs(tl, "qubit")
    check("second pulse follows the first", close(qs[1].t0, 40))
    check("elements are independent", close(segs(tl, "res")[0].t0, 0))
    check("wait is in clock cycles", close(qs[2].t0, 400),
          f"(pulse 3 at {qs[2].t0} ns)")
    check("align takes the max cursor",
          close(max(tl.cursors["qubit"], tl.cursors["res"]), 440))


def test_align_all():
    print("  [align() with no arguments]")
    with q.program() as p:
        q.play("long", "qubit")          # 1000 ns
        q.play("step", "gate")           # 16 ns
        q.align()
        q.play("step", "gate")
    tl = run(p)
    check("bare align covers every element used",
          close(segs(tl, "gate")[1].t0, 1000))

    with q.program() as p2:                  # res is first used AFTER the align
        q.play("long", "qubit")
        q.align()
        q.play("readout", "res")
    tl2 = run(p2)
    check("bare align also covers elements used later in the program",
          close(segs(tl2, "res")[0].t0, 1000),
          f"(readout at {segs(tl2, 'res')[0].t0} ns)")


def test_amp_and_duration():
    print("  [amp() and duration=]")
    with q.program() as p:
        a = q.declare(q.fixed, value=0.5)
        q.play("pi" * q.amp(a), "qubit")
        q.play("pi", "qubit", duration=100)          # 100 cycles = 400 ns
        q.play("pi", "qubit", truncate=5)            # 5 cycles = 20 ns
    tl = run(p)
    s = segs(tl, "qubit")
    check("amp() scale comes from the variable", close(s[0].amp, 0.5))
    check("duration= overrides the config length", close(s[1].duration, 400))
    check("truncate= shortens the pulse", close(s[2].duration, 20))
    check("peak follows the amp scale",
          close(tl.peak("qubit", "I"), 0.2), f"({tl.peak('qubit', 'I')})")


def test_measure_window():
    print("  [measurement window]")
    with q.program() as p:
        I = q.declare(q.fixed)
        q.measure("readout", "res", None, q.demod.full("cos", I, "out1"))
    tl = run(p)
    a = tl.acquisitions[0]
    check("ADC window starts at tof - smearing", close(a.t0, 24 - 8))
    check("ADC window ends at length + tof + smearing", close(a.t1, 400 + 24 + 8))
    check("the element is busy for the pulse length",
          close(segs(tl, "res")[0].t1, 400))


def test_loops():
    print("  [loop unrolling]")
    with q.program() as p:
        n = q.declare(int)
        with q.for_(n, 0, n < 10, n + 1):
            q.play("pi", "qubit")
    tl = run(p, iterations=3)
    check("unrolls exactly --iterations times", len(segs(tl, "qubit")) == 3)
    check("records how many iterations were skipped",
          tl.breaks and tl.breaks[0].skipped == 7,
          f"(skipped={tl.breaks[0].skipped if tl.breaks else None})")

    with q.program() as p2:
        a = q.declare(q.fixed)
        with q.for_each_(a, [0.1, 0.4, 0.9]):
            q.play("pi" * q.amp(a), "qubit")
    tl2 = run(p2, iterations=2)
    amps = [s.amp for s in segs(tl2, "qubit")]
    check("for_each_ uses the concrete sweep values", amps == [0.1, 0.4],
          f"({amps})")
    check("for_each_ reports the rest", tl2.breaks[0].skipped == 1)

    with q.program() as p3:
        n = q.declare(int)
        with q.for_(n, 0, n < 3, n + 1):
            q.play("pi", "qubit")
    tl3 = run(p3, iterations=10)
    check("a loop shorter than the limit runs to completion and does not break",
          len(segs(tl3, "qubit")) == 3 and not tl3.breaks)


def test_runtime_branch():
    print("  [run-time-dependent control flow]")
    with q.program() as p:
        I = q.declare(q.fixed)
        with q.for_(q.declare(int), 0, q.declare(int) < 1, 1):
            pass
        q.measure("readout", "res", None, q.demod.full("cos", I, "out1"))
        with q.if_(I > 0.001):
            q.play("pi", "qubit")
        with q.else_():
            q.play("long", "qubit")
    tl_auto = run(p)
    check("a measured value taints the branch",
          any(m.kind == "branch" for m in tl_auto.markers))
    check("--branch auto takes the branch as evaluated (I=0)",
          [s.op for s in segs(tl_auto, "qubit")] == ["long"])
    tl_then = run(p, branch="then")
    check("--branch then draws the other side",
          [s.op for s in segs(tl_then, "qubit")] == ["pi"])
    check("segments inside a run-time branch are marked",
          all(s.runtime for s in segs(tl_then, "qubit")))


def test_sticky_and_ramp():
    print("  [sticky elements]")
    with q.program() as p:
        q.play("step", "gate")
        q.play("step", "gate")
        q.play("step", "gate")
        q.wait(25, "gate")            # 100 ns with nothing played
        q.ramp_to_zero("gate")
    tl = run(p)
    levels = [round(L.level, 6) for L in tl.levels if L.element == "gate"]
    check("each step adds to the held level", levels[:3] == [0.02, 0.04, 0.06],
          f"({levels})")
    check("ramp_to_zero returns to 0", close(levels[-1], 0.0))
    check("the lane scale is the held level, not level+pulse",
          close(tl.peak("gate", "single"), 0.06),
          f"({tl.peak('gate', 'single')})")
    check("the held level persists through an idle gap",
          close(tl.range_over("gate", "single", 100, 104)[1], 0.06),
          f"({tl.range_over('gate', 'single', 100, 104)})")


def test_frame_and_frequency():
    print("  [frame and frequency updates]")
    with q.program() as p:
        q.play("pi", "qubit")
        q.update_frequency("qubit", 25e6)
        q.frame_rotation_2pi(0.25, "qubit")
        q.play("pi", "qubit")
        q.reset_frame("qubit")
        q.play("pi", "qubit")
    tl = run(p)
    s = segs(tl, "qubit")
    check("IF applies from the update onwards",
          close(s[0].freq, 100e6) and close(s[1].freq, 25e6))
    check("frame_rotation_2pi(0.25) is pi/2", close(s[1].phase, 1.5707963, 1e-5))
    check("reset_frame clears the phase", close(s[2].phase, 0.0))
    check("updates are neither pulses nor delays", close(s[1].t0, 40))


def test_expressions():
    print("  [expression evaluation]")
    with q.program() as p:
        n = q.declare(int, value=3)
        a = q.declare(q.fixed)
        q.assign(a, q.Math.cos(0.0) * 0.5)
        q.play("pi" * q.amp(a), "qubit")
        q.assign(a, q.Cast.mul_int_by_fixed(n, 0.25))
        q.play("pi" * q.amp(a), "qubit")
    tl = run(p)
    amps = [s.amp for s in segs(tl, "qubit")]
    check("Math and arithmetic evaluate", close(amps[0], 0.5))
    check("Cast.mul_int_by_fixed evaluates", close(amps[1], 0.0),
          "(3 * 0.25 truncated to int, as on hardware)")


def test_examples_end_to_end():
    print("  [examples end to end]")
    here = os.path.dirname(os.path.abspath(__file__))
    ex = os.path.join(here, "..", "examples")
    for name in sorted(os.listdir(ex)):
        if not name.endswith(".py") or name == "configuration.py":
            continue
        res = load_script(os.path.join(ex, name))
        tl = Interpreter(Machine(res.config), RunConfig(max_iterations=2)).run(
            res.program)
        svg = render_svg(tl, title=name)
        txt = render_text(tl, width=80)
        ok = True
        try:
            ET.fromstring(svg)
        except ET.ParseError as e:
            ok = False
            print(f"      {e}")
        check(f"{name}", bool(tl.segments) and ok and bool(txt),
              f"({tl.summary()})")


def main():
    for suite in (test_cursors, test_align_all, test_amp_and_duration,
                  test_measure_window, test_loops, test_runtime_branch,
                  test_sticky_and_ramp, test_frame_and_frequency,
                  test_expressions, test_examples_end_to_end):
        suite()
    print()
    if failures:
        print(f"{len(failures)} FAILURES")
        for f in failures:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
