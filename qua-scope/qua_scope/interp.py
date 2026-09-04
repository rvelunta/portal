"""Interpret a traced QUA program into a Timeline.

Two things happen here.

*Unrolling*: the IR is a tree with loops, a timeline is flat. Loops are executed
with concrete variable values -- so a swept amplitude really is 0.1, 0.2, ... in
the picture -- up to `RunConfig.max_iterations`, after which a Break records how
many iterations were left out. Rendering 1000 averaging iterations helps nobody;
rendering two and saying "x998 more" does.

*Scheduling*: QUA gives every element its own cursor. A statement advances only
the elements it names, `align()` snaps a set of cursors to the latest of them,
and `wait()` pushes cursors forward. That is the entire timing model, plus:
pulse length comes from the config (or `duration=` in clock cycles), sticky
elements hold their last level, and a measurement opens an ADC window at
time_of_flight, widened by smearing.

Values that only exist at run time (demodulation results, `Random`, input
streams) are tainted: they evaluate to 0 but any branch taken on them is marked
runtime-dependent, and the renderer says so rather than pretending.
"""

import math
from dataclasses import dataclass

from . import ir
from .machine import Machine, Waveform
from .timeline import (CLOCK_NS, Acquisition, Break, LevelChange, Marker,
                       Segment, Timeline, fmt_freq, fmt_time)


@dataclass
class RunConfig:
    max_iterations: int = 2        # unrolled iterations per loop
    max_segments: int = 20000      # hard budget, protects against blowups
    max_time_ns: float = 0.0       # 0 = no limit
    branch: str = "auto"           # 'auto' | 'then' | 'else' for runtime ifs


class _Stop(Exception):
    """Budget exhausted; the timeline so far is still valid."""


class Interpreter:
    def __init__(self, machine, cfg=None):
        self.m = machine if isinstance(machine, Machine) else Machine(machine)
        self.cfg = cfg or RunConfig()
        self.tl = Timeline(self.m)
        self.env = {}              # QuaVar name -> value (or list)
        self.tainted = set()       # names holding run-time-only values
        self.t = {}                # element -> cursor (ns)
        self.freq = {}             # element -> IF (Hz)
        self.phase = {}            # element -> frame phase (rad)
        self.level = {}            # (element, channel) -> DC level (V)
        self.iteration = ()        # loop index path, for labelling
        self.runtime_depth = 0     # >0 inside a run-time-dependent branch
        self.known_elements = []   # every element the program mentions

    # ------------------------------------------------------------------ run
    def run(self, program):
        # A bare align() covers every element the PROGRAM mentions, including
        # ones first played later on -- QUA is compiled whole, so collect them
        # up front rather than only the ones seen so far.
        self.known_elements = _elements_in(program.body)
        for v in program.variables:
            self.env[v.name] = list(v.init) if isinstance(v.init, list) else (
                0 if v.init is None else v.init)
            if v.is_input_stream:
                self.tainted.add(v.name)
        try:
            self._body(program.body)
        except _Stop as e:
            self.tl.warnings.append(str(e))
        self.tl.cursors = dict(self.t)
        self.tl.duration = max([self.tl.duration] + list(self.t.values()))
        self.tl.warnings.extend(self.m.warnings)
        self.tl.stats = {
            "segments": len(self.tl.segments),
            "elements": len(self.tl.elements),
            "measurements": len(self.tl.acquisitions),
        }
        return self.tl

    # ------------------------------------------------------------ evaluation
    def _ev(self, e):
        """Evaluate an expression -> (value, tainted)."""
        if e is None:
            return None, False
        if isinstance(e, ir.Lit):
            return e.value, False
        if isinstance(e, ir.QuaVar):
            return self.env.get(e.name, 0), e.name in self.tainted
        if isinstance(e, ir.Index):
            base, t1 = self._ev(e.base)
            idx, t2 = self._ev(e.index)
            try:
                return base[int(idx)], t1 or t2
            except Exception:
                return 0, True
        if isinstance(e, ir.UnOp):
            v, t = self._ev(e.operand)
            if e.op == "-":
                return -v, t
            return (not v) if e.op == "~" else v, t
        if isinstance(e, ir.BinOp):
            a, t1 = self._ev(e.lhs)
            b, t2 = self._ev(e.rhs)
            return _binop(e.op, a, b), t1 or t2
        if isinstance(e, ir.Call):
            return self._call(e)
        if isinstance(e, ir.Expr):
            return 0, True
        return e, False            # a plain Python value

    def _call(self, e):
        args, tainted = [], False
        for a in e.args:
            v, t = self._ev(a)
            args.append(v)
            tainted = tainted or t
        fn = e.fn.split(".")[-1]
        try:
            if fn in _MATH1 and len(args) == 1:
                return _MATH1[fn](args[0]), tainted
            if fn in _MATH2 and len(args) == 2:
                return _MATH2[fn](args[0], args[1]), tainted
            if fn == "array_length":
                return len(args[0]), tainted
            if fn.startswith("to_int") or fn == "mul_int_by_fixed":
                return (int(args[0]) if len(args) == 1
                        else int(args[0] * args[1])), tainted
            if fn in ("to_fixed", "mul_fixed_by_int", "unsafe_cast_fixed"):
                return (float(args[0]) if len(args) == 1
                        else float(args[0] * args[1])), tainted
            if fn == "cond":
                return (args[1] if args[0] else args[2]), tainted
            if fn.startswith("rand"):
                return 0, True     # run-time random: value unknown here
        except Exception:
            pass
        return 0, True

    # ------------------------------------------------------------- execution
    def _body(self, body):
        for s in body:
            self._exec(s)

    def _exec(self, s):
        if len(self.tl.segments) > self.cfg.max_segments:
            raise _Stop(f"stopped after {self.cfg.max_segments} pulses "
                        f"(--max-segments)")
        if self.cfg.max_time_ns and min(self.t.values() or [0]) > self.cfg.max_time_ns:
            raise _Stop(f"stopped at {fmt_time(self.cfg.max_time_ns)} (--tmax)")

        h = _DISPATCH.get(type(s))
        if h is not None:
            h(self, s)

    # -- statements ---------------------------------------------------------
    def _cursor(self, el):
        return self.t.setdefault(el, 0.0)

    def _all_elements(self):
        els = list(self.t)
        for e in list(self.tl.elements) + list(self.known_elements):
            if e not in els:
                els.append(e)
        return els

    def _resolve_amp(self, pulse_ref):
        """amp() scale actually applied, plus a label if it came from a variable."""
        if pulse_ref.amp is None:
            return 1.0, ""
        spec = pulse_ref.amp
        v, tainted = self._ev(spec.values[0])
        try:
            v = float(v)
        except (TypeError, ValueError):
            v, tainted = 1.0, True
        label = ""
        if not isinstance(spec.values[0], ir.Lit):
            label = f"*amp({_short(spec.values[0])}={v:.3g})"
        elif abs(v - 1.0) > 1e-12:
            label = f"*{v:.3g}"
        if len(spec.values) == 4:
            label += " [matrix]"
        return v, label

    def _waves(self, pulse):
        out = {}
        for ch, wfname in (pulse.waveforms or {}).items():
            out[ch] = self.m.waveform(wfname)
        return out

    def do_play(self, s, kind="play"):
        el = self.m.element(s.element)
        t0 = self._cursor(s.element)
        amp, amp_label = self._resolve_amp(s.pulse)
        runtime = self.runtime_depth > 0

        if s.pulse.ramp_rate is not None:
            rate, _t = self._ev(s.pulse.ramp_rate)
            dur, _ = self._ev(s.duration)
            length = int(dur or 1) * CLOCK_NS
            n = min(length, 256)
            wf = Waveform("ramp", "arb",
                          samples=[rate * (i * length / n) for i in range(n)])
            waves = {ch: wf for ch in el.channels}
            pulse_name, op = "ramp", f"ramp({rate:.3g} V/ns)"
        else:
            pulse = self.m.pulse_for(s.element, s.pulse.operation)
            length = pulse.length
            if s.duration is not None:
                d, tainted = self._ev(s.duration)
                if d:
                    length = int(d) * CLOCK_NS
                    runtime = runtime or tainted
            if s.truncate is not None:
                d, _ = self._ev(s.truncate)
                if d:
                    length = min(length, int(d) * CLOCK_NS)
            waves = self._waves(pulse)
            pulse_name, op = pulse.name, s.pulse.operation or pulse.name

        if s.condition is not None:                 # play(..., condition=...)
            cond, tainted = self._ev(s.condition)
            runtime = runtime or tainted
            if not cond and not tainted:
                self.t[s.element] = t0 + length     # slot still consumed
                return

        seg = Segment(
            element=s.element, kind=kind, t0=t0, t1=t0 + length,
            op=op, pulse=pulse_name, waves=waves, amp=amp, amp_label=amp_label,
            freq=self.freq.get(s.element, el.intermediate_frequency),
            phase=self.phase.get(s.element, 0.0), runtime=runtime,
            loc=str(s.loc), iteration=self.iteration,
        )
        self.tl.add(seg)
        self.t[s.element] = t0 + length

        if el.sticky:                                # level accumulates and holds
            for ch, wf in waves.items():
                last = (wf.value if wf.kind == "const"
                        else (wf.samples[-1] if wf.samples else 0.0)) * amp
                key = (s.element, ch)
                self.level[key] = self.level.get(key, 0.0) + last
                self.tl.add(LevelChange(s.element, ch, t0 + length,
                                        self.level[key], "sticky hold"))
        return seg

    def do_measure(self, s):
        el = self.m.element(s.element)
        seg = self.do_play(ir.Play(s.pulse, s.element, duration=s.duration,
                                   loc=s.loc), kind="measure")
        for tgt in s.targets:                        # demod results are unknown
            name = getattr(tgt, "name", None) or getattr(
                getattr(tgt, "base", None), "name", None)
            if name:
                self.tainted.add(name)
        if seg is not None:
            self.tl.add(Acquisition(
                element=s.element,
                t0=seg.t0 + el.time_of_flight - el.smearing,
                t1=seg.t1 + el.time_of_flight + el.smearing,
                label="; ".join(s.demods) or "measure", loc=str(s.loc)))

    def do_wait(self, s):
        d, tainted = self._ev(s.duration)
        try:
            ns = int(d) * CLOCK_NS
        except (TypeError, ValueError):
            ns, tainted = CLOCK_NS, True
        els = s.elements or self._all_elements()
        if not s.elements:
            self.tl.warnings.append(
                f"{s.loc}: wait() with no element -- applied to all")
        starts = {e: self._cursor(e) for e in els}
        for e in els:
            self.t[e] = starts[e] + ns
            self.tl.touch(e)
        if ns >= 100 and starts:
            self.tl.add(Marker("wait", min(starts.values()), list(els),
                               f"wait {fmt_time(ns)}"
                               + (" (run-time)" if tainted else ""), str(s.loc)))

    def do_align(self, s):
        els = s.elements or self._all_elements()
        if not els:
            return
        tmax = max(self._cursor(e) for e in els)
        for e in els:
            self.t[e] = tmax
            self.tl.touch(e)
        self.tl.add(Marker("align", tmax, list(els),
                           "align" + ("" if s.elements else " (all)"), str(s.loc)))

    def do_assign(self, s):
        val, tainted = self._ev(s.value)
        tgt = s.target
        if isinstance(tgt, ir.Index):
            base = self.env.get(tgt.base.name)
            idx, ti = self._ev(tgt.index)
            if isinstance(base, list):
                try:
                    base[int(idx)] = val
                except Exception:
                    pass
            name = tgt.base.name
            tainted = tainted or ti
        else:
            name = getattr(tgt, "name", None)
            if name:
                self.env[name] = val
        if name:
            self.tainted.discard(name)
            if tainted:
                self.tainted.add(name)

    def do_update_frequency(self, s):
        v, tainted = self._ev(s.frequency)
        scale = {"Hz": 1, "mHz": 1e-3, "uHz": 1e-6, "nHz": 1e-9}.get(s.units, 1)
        try:
            f = float(v) * scale
        except (TypeError, ValueError):
            f, tainted = 0.0, True
        self.freq[s.element] = f
        self.tl.touch(s.element)
        self.tl.add(Marker("freq", self._cursor(s.element), [s.element],
                           f"IF -> {fmt_freq(f)}" + (" (run-time)" if tainted else ""),
                           str(s.loc)))

    def do_frame_rotation(self, s):
        a, tainted = self._ev(s.angle)
        try:
            a = float(a)
        except (TypeError, ValueError):
            a, tainted = 0.0, True
        for e in s.elements or self._all_elements():
            self.phase[e] = self.phase.get(e, 0.0) + a
            self.tl.touch(e)
            self.tl.add(Marker("frame", self._cursor(e), [e],
                               f"frame +{a / math.pi:.3g}pi"
                               + (" (run-time)" if tainted else ""), str(s.loc)))

    def do_reset_frame(self, s):
        for e in s.elements or self._all_elements():
            self.phase[e] = 0.0
            self.tl.add(Marker("frame", self._cursor(e), [e], "reset frame",
                               str(s.loc)))

    def do_reset_phase(self, s):
        self.phase[s.element] = 0.0
        self.tl.touch(s.element)
        self.tl.add(Marker("phase", self._cursor(s.element), [s.element],
                           "reset phase", str(s.loc)))

    def do_set_dc_offset(self, s):
        v, tainted = self._ev(s.offset)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v, tainted = 0.0, True
        ch = s.element_input if s.element_input in ("I", "Q") else "single"
        self.level[(s.element, ch)] = v
        self.tl.touch(s.element)
        self.tl.add(LevelChange(s.element, ch, self._cursor(s.element), v,
                                "set_dc_offset"))
        self.tl.add(Marker("dc", self._cursor(s.element), [s.element],
                           f"dc {ch} = {v:+.3g} V", str(s.loc)))

    def do_ramp_to_zero(self, s):
        el = self.m.element(s.element)
        d, _ = self._ev(s.duration)
        length = int(d or 1) * CLOCK_NS
        t0 = self._cursor(s.element)
        for ch in el.channels:
            key = (s.element, ch)
            start = self.level.get(key, 0.0)
            if start:
                n = min(length, 128)
                wf = Waveform("ramp0", "arb",
                              samples=[-start * i / max(1, n - 1) for i in range(n)])
                self.tl.add(Segment(element=s.element, kind="ramp", t0=t0,
                                    t1=t0 + length, op="ramp_to_zero",
                                    pulse="ramp_to_zero", waves={ch: wf},
                                    loc=str(s.loc), iteration=self.iteration,
                                    runtime=self.runtime_depth > 0))
            self.level[key] = 0.0
            self.tl.add(LevelChange(s.element, ch, t0 + length, 0.0,
                                    "ramp_to_zero"))
        self.t[s.element] = t0 + length

    def do_pause(self, s):
        t = max(self.t.values() or [0.0])
        for e in self._all_elements():
            self.t[e] = t
        self.tl.add(Marker("pause", t, self._all_elements(),
                           "pause (waits for the host)", str(s.loc)))

    def do_wait_for_trigger(self, s):
        t = self._cursor(s.element)
        self.tl.touch(s.element)
        self.tl.add(Marker("trigger", t, [s.element],
                           "wait for trigger (unknown delay)", str(s.loc)))
        if s.pulse is not None:
            self.do_play(ir.Play(s.pulse, s.element, loc=s.loc), kind="trigger")

    # -- blocks -------------------------------------------------------------
    def _loop_break(self, skipped, label):
        t = max(self.t.values() or [0.0])
        self.tl.add(Break(t, skipped, label))

    def do_for(self, s):
        var_name = getattr(s.var, "name", None)
        init, _ = self._ev(s.init)
        if var_name:
            self.env[var_name] = init
            self.tainted.discard(var_name)
        outer, n = self.iteration, 0
        while True:
            cond, tainted = (True, False) if s.cond is None else self._ev(s.cond)
            if not cond:
                break
            if n >= self.cfg.max_iterations:
                self._loop_break(self._remaining_for(s, var_name),
                                 f"loop at {s.loc}")
                break
            self.iteration = outer + (n,)
            self._body(s.body)
            n += 1
            if var_name and s.update is not None:
                v, t = self._ev(s.update)
                self.env[var_name] = v
                if t:
                    self.tainted.add(var_name)
            elif s.update is None:
                break
        self.iteration = outer

    def _remaining_for(self, s, var_name):
        """How many iterations we skipped -- exact when the body does not touch
        the loop variable, otherwise None (rendered as '...')."""
        if not var_name or s.update is None or _assigns(s.body, var_name):
            return None
        saved = self.env.get(var_name)
        n = 0
        try:
            while n < 10_000_000:
                cond, tainted = self._ev(s.cond)
                if tainted or not cond:
                    break
                n += 1
                self.env[var_name], _ = self._ev(s.update)
        finally:
            self.env[var_name] = saved
        return n if n < 10_000_000 else None

    def do_for_each(self, s):
        names = [getattr(v, "name", None) for v in s.vars]
        total = min(len(v) for v in s.values) if s.values else 0
        outer = self.iteration
        for i in range(total):
            if i >= self.cfg.max_iterations:
                self._loop_break(total - i, f"for_each_ at {s.loc}")
                break
            for name, vals in zip(names, s.values):
                if name:
                    self.env[name] = vals[i]
                    self.tainted.discard(name)
            self.iteration = outer + (i,)
            self._body(s.body)
        self.iteration = outer

    def do_while(self, s):
        outer, n = self.iteration, 0
        while True:
            cond, tainted = self._ev(s.cond)
            if tainted:
                if n >= self.cfg.max_iterations:
                    self._loop_break(None, f"while_ at {s.loc} (run-time bound)")
                    break
            elif not cond:
                break
            if n >= self.cfg.max_iterations:
                self._loop_break(None, f"while_ at {s.loc}")
                break
            self.iteration = outer + (n,)
            self.runtime_depth += bool(tainted)
            self._body(s.body)
            self.runtime_depth -= bool(tainted)
            n += 1
        self.iteration = outer

    def do_infinite_loop(self, s):
        outer = self.iteration
        for i in range(max(1, self.cfg.max_iterations)):
            self.iteration = outer + (i,)
            self._body(s.body)
        self.iteration = outer
        self._loop_break(None, "infinite_loop_ (repeats forever)")

    def do_if(self, s):
        branches = [(s.cond, s.body)] + list(s.elifs)
        for cond, body in branches:
            val, tainted = self._ev(cond)
            take = bool(val)
            if tainted:
                take = {"then": True, "else": False}.get(self.cfg.branch, take)
                self.tl.add(Marker(
                    "branch", max(self.t.values() or [0.0]), [],
                    f"run-time branch: {'taken' if take else 'not taken'} "
                    f"({_short(cond)})", str(s.loc)))
            if take:
                self.runtime_depth += bool(tainted)
                self._body(body)
                self.runtime_depth -= bool(tainted)
                return
        if s.orelse is not None:
            self._body(s.orelse)

    def do_switch(self, s):
        val, tainted = self._ev(s.expr)
        for cval, body in s.cases:
            cv, _ = self._ev(ir.lit(cval))
            if (cv == val) or tainted:
                if tainted:
                    self.tl.add(Marker("branch", max(self.t.values() or [0.0]),
                                       [], f"run-time switch: case {cv}",
                                       str(s.loc)))
                self.runtime_depth += bool(tainted)
                self._body(body)
                self.runtime_depth -= bool(tainted)
                return
        if s.default is not None:
            self._body(s.default)

    def do_strict(self, s):
        self._body(s.body)

    def do_nothing(self, s):
        pass


# --------------------------------------------------------------------- helpers

def _binop(op, a, b):
    try:
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return a / b if b else 0
        if op == "//": return a // b if b else 0
        if op == "%": return a % b if b else 0
        if op == "<": return a < b
        if op == "<=": return a <= b
        if op == ">": return a > b
        if op == ">=": return a >= b
        if op == "==": return a == b
        if op == "!=": return a != b
        if op == "&": return bool(a) and bool(b)
        if op == "|": return bool(a) or bool(b)
        if op == "^": return bool(a) != bool(b)
        if op == "<<": return int(a) << int(b)
        if op == ">>": return int(a) >> int(b)
    except Exception:
        return 0
    return 0


_MATH1 = {"cos": math.cos, "sin": math.sin, "tan": math.tan, "exp": math.exp,
          "ln": math.log, "log10": math.log10, "log2": math.log2,
          "sqrt": math.sqrt, "abs": abs, "inv": lambda x: 1 / x if x else 0,
          "cos2pi": lambda x: math.cos(2 * math.pi * x),
          "sin2pi": lambda x: math.sin(2 * math.pi * x),
          "msb": lambda x: int(math.floor(math.log2(abs(x)))) if x else 0}
_MATH2 = {"pow": lambda a, b: a ** b, "max": max, "min": min,
          "div": lambda a, b: a / b if b else 0,
          "log": lambda a, b: math.log(a, b)}


def _elements_in(body, out=None):
    """Element names mentioned anywhere in a statement tree, in first-use order."""
    out = [] if out is None else out
    for s in body:
        for name in ([getattr(s, "element", None)] + list(getattr(s, "elements", []) or [])):
            if isinstance(name, str) and name not in out:
                out.append(name)
        for attr in ("body", "orelse", "default"):
            sub = getattr(s, attr, None)
            if sub:
                _elements_in(sub, out)
        for _, sub in (getattr(s, "elifs", None) or []):
            _elements_in(sub, out)
        for _, sub in (getattr(s, "cases", None) or []):
            _elements_in(sub, out)
    return out


def _assigns(body, name):
    """Does this block assign `name`? (loop-count estimation guard)"""
    for s in body:
        if isinstance(s, ir.Assign):
            tgt = s.target
            n = getattr(tgt, "name", None) or getattr(
                getattr(tgt, "base", None), "name", None)
            if n == name:
                return True
        for attr in ("body", "orelse", "default"):
            sub = getattr(s, attr, None)
            if sub and _assigns(sub, name):
                return True
        for _, sub in (getattr(s, "elifs", None) or []):
            if _assigns(sub, name):
                return True
        for _, sub in (getattr(s, "cases", None) or []):
            if _assigns(sub, name):
                return True
    return False


def _short(e, n=32):
    s = repr(e)
    return s if len(s) <= n else s[:n - 1] + "..."


_DISPATCH = {
    ir.Play: Interpreter.do_play,
    ir.Measure: Interpreter.do_measure,
    ir.Wait: Interpreter.do_wait,
    ir.Align: Interpreter.do_align,
    ir.Assign: Interpreter.do_assign,
    ir.UpdateFrequency: Interpreter.do_update_frequency,
    ir.FrameRotation: Interpreter.do_frame_rotation,
    ir.ResetFrame: Interpreter.do_reset_frame,
    ir.ResetPhase: Interpreter.do_reset_phase,
    ir.SetDCOffset: Interpreter.do_set_dc_offset,
    ir.RampToZero: Interpreter.do_ramp_to_zero,
    ir.Pause: Interpreter.do_pause,
    ir.WaitForTrigger: Interpreter.do_wait_for_trigger,
    ir.For: Interpreter.do_for,
    ir.ForEach: Interpreter.do_for_each,
    ir.While: Interpreter.do_while,
    ir.InfiniteLoop: Interpreter.do_infinite_loop,
    ir.If: Interpreter.do_if,
    ir.Switch: Interpreter.do_switch,
    ir.StrictTiming: Interpreter.do_strict,
    ir.Save: Interpreter.do_nothing,
    ir.UpdateCorrection: Interpreter.do_nothing,
    ir.Unknown: Interpreter.do_nothing,
}


def interpret(program, config, cfg=None):
    """Convenience: traced program + QM config dict -> Timeline."""
    return Interpreter(config, cfg).run(program)
