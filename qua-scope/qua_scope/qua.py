"""A tracing stand-in for `qm.qua`.

The loader installs this module as `qm.qua`, so an unmodified QUA script that
does `from qm.qua import *` runs here instead of against the real SDK. Nothing
is compiled or sent anywhere: every statement is appended to an IR tree
(ir.Program), and `with for_(...)` / `with if_(...)` push a nested block.

Trace time vs. run time is the whole trick. `n < 100` cannot be evaluated while
tracing -- `n` has no value -- so QuaVar operators build expression trees that
interp.py evaluates later, once per unrolled iteration.

If the real `qm` package is importable it is NOT used: we always trace, because
we want the program tree, not a serialized job.
"""

import math
import sys
from contextlib import contextmanager

from .ir import (
    Align, AmpSpec, Assign, BinOp, Call, For, ForEach, If, InfiniteLoop, Index,
    Lit, Loc, Measure, Pause, Play, Program, PulseRef, QuaVar, RampToZero,
    ResetFrame, ResetPhase, Save, SetDCOffset, StrictTiming, Switch, Unknown,
    UpdateCorrection, UpdateFrequency, Wait, WaitForTrigger, While,
    FrameRotation, lit,
)

__all__ = [
    "program", "declare", "declare_stream", "declare_input_stream", "fixed",
    "play", "wait", "align", "measure", "assign", "save", "pause",
    "update_frequency", "update_correction", "frame_rotation",
    "frame_rotation_2pi", "reset_frame", "reset_phase", "reset_if_phase",
    "set_dc_offset", "ramp_to_zero", "wait_for_trigger", "advance_input_stream",
    "for_", "for_each_", "if_", "elif_", "else_", "while_", "infinite_loop_",
    "switch_", "case_", "default_", "strict_timing_", "port_condition",
    "stream_processing", "amp", "ramp", "demod", "dual_demod", "integration",
    "time_tagging", "counting", "Math", "Cast", "Util", "Random", "L",
    "traced_programs", "reset_trace",
]


# ------------------------------------------------------------------ trace state

class _Builder:
    """Accumulates statements into the innermost open block."""

    def __init__(self, source="?"):
        self.program = Program(source=source)
        self.stack = [self.program.body]
        self.last_if = {}          # id(body list) -> If node, for elif_/else_
        self.suppress = 0          # >0 inside stream_processing: drop statements
        self.n_vars = 0
        self.names = set()         # declared variable names, kept unique

    def emit(self, stmt):
        if self.suppress:
            return stmt
        body = self.stack[-1]
        body.append(stmt)
        if isinstance(stmt, If):
            self.last_if[id(body)] = stmt
        elif not isinstance(stmt, Unknown):
            self.last_if.pop(id(body), None)
        return stmt

    def push(self, body):
        self.stack.append(body)

    def pop(self):
        self.stack.pop()


_stack: list = []            # open programs (nesting is unusual but harmless)
traced_programs: list = []   # every program traced since the last reset


def reset_trace():
    _stack.clear()
    traced_programs.clear()


def _b():
    if not _stack:
        raise RuntimeError(
            "QUA statement outside a program -- wrap the sequence in "
            "`with program() as prog:`"
        )
    return _stack[-1]


def _loc():
    """Innermost stack frame that is not part of qua_scope."""
    import os
    f = sys._getframe(1)
    here = os.path.dirname(os.path.abspath(__file__))
    while f is not None:
        name = f.f_code.co_filename
        if not name.startswith(here) and "importlib" not in name:
            return Loc(name, f.f_lineno)
        f = f.f_back
    return Loc()


def _elname(e):
    """Elements are strings in QUA; accept objects that stringify sanely."""
    if isinstance(e, str):
        return e
    for attr in ("name", "element", "id"):
        v = getattr(e, attr, None)
        if isinstance(v, str):
            return v
    return str(e)


# ------------------------------------------------------------------ program

class _ProgramScope:
    def __init__(self):
        self._builder = None

    def __enter__(self):
        src = _loc().file
        self._builder = _Builder(src)
        _stack.append(self._builder)
        return self._builder.program

    def __exit__(self, *exc):
        _stack.pop()
        traced_programs.append(self._builder.program)
        return False

    # `qmm.execute(prog)` and friends poke at these; keep them harmless.
    def __getattr__(self, item):
        raise AttributeError(item)


def program():
    return _ProgramScope()


# ------------------------------------------------------------------ types/vars

class fixed:
    """QUA's 4.28 fixed-point type. Used only as a tag in `declare()`."""


_TYPE_NAMES = {int: "int", bool: "bool", float: "fixed", fixed: "fixed"}


def _declared_name(fallback):
    """Recover the user's variable name from the source line of the declare(),
    so labels read `amp(a=0.25)` and not `amp(v2=0.25)`."""
    import linecache
    import re
    loc = _loc()
    line = linecache.getline(loc.file, loc.line).strip()
    m = re.match(r"([A-Za-z_]\w*)\s*=\s*declare", line)
    return m.group(1) if m else fallback


def declare(t=int, value=None, size=None):
    b = _b()
    tname = _TYPE_NAMES.get(t, getattr(t, "__name__", str(t)))
    b.n_vars += 1
    name = _declared_name(f"v{b.n_vars}")
    if name in b.names:                    # same source name declared twice
        name = f"{name}#{b.n_vars}"
    b.names.add(name)
    n = size
    if n is None and isinstance(value, (list, tuple)):
        n = len(value)
    var = QuaVar(name=name, type=tname, size=n,
                 init=list(value) if isinstance(value, (list, tuple)) else value)
    b.program.variables.append(var)
    return var


def declare_input_stream(t=int, name=None, value=None, size=None):
    var = declare(t, value=value, size=size)
    var.is_input_stream = True
    if name:
        var.name = str(name)
    return var


def advance_input_stream(var):
    _b().emit(Unknown("advance_input_stream", getattr(var, "name", ""), loc=_loc()))


class _Stream:
    """A result stream. Stream-processing chains are recorded loosely: the
    scope diagram cares about acquisition timing, not about the DAG."""

    def __init__(self, tag="stream", adc_trace=False):
        self.tag = tag
        self.adc_trace = adc_trace
        self.ops = []

    def __getattr__(self, op):
        if op.startswith("_"):
            raise AttributeError(op)

        def chain(*a, **k):
            self.ops.append((op, a, k))
            if op == "save" or op == "save_all":
                self.tag = a[0] if a else self.tag
            return self
        return chain

    def __repr__(self):
        return f"<stream {self.tag}>"


def declare_stream(adc_trace=False, **_):
    b = _b()
    s = _Stream(f"s{len(b.program.streams) + 1}", adc_trace)
    b.program.streams.append(s)
    return s


def assign(target, value):
    return _b().emit(Assign(target, lit(value), loc=_loc()))


def save(source, stream_or_tag):
    return _b().emit(Save(source, stream_or_tag, loc=_loc()))


def L(value):
    """QUA literal helper (`L(3)`); we keep plain Python values."""
    return value


# ------------------------------------------------------------------ pulse specs

class amp:
    """`"pulse" * amp(v)` scales the played waveform. 1 or 4 arguments."""

    def __init__(self, *values):
        if len(values) not in (1, 4):
            raise ValueError("amp() takes 1 or 4 values")
        self.spec = AmpSpec([lit(v) for v in values])

    def __rmul__(self, other):     # "pulse" * amp(a)
        return PulseRef(operation=_elname(other), amp=self.spec)

    __mul__ = __rmul__


class ramp:
    """`play(ramp(rate), el, duration=...)` -- a linear voltage ramp, V/ns."""

    def __init__(self, rate):
        self.rate = lit(rate)


def _pulse_ref(p):
    if isinstance(p, PulseRef):
        return p
    if isinstance(p, ramp):
        return PulseRef(ramp_rate=p.rate)
    return PulseRef(operation=_elname(p))


# ------------------------------------------------------------------ statements

def play(pulse, element, duration=None, condition=None, chirp=None,
         truncate=None, timestamp_stream=None, continue_chirp=False, **_):
    return _b().emit(Play(
        _pulse_ref(pulse), _elname(element), duration=duration,
        condition=condition, truncate=truncate, chirp=chirp,
        timestamp_stream=timestamp_stream, loc=_loc()))


def wait(duration, *elements, **_):
    return _b().emit(Wait(duration, [_elname(e) for e in elements], loc=_loc()))


def align(*elements):
    return _b().emit(Align([_elname(e) for e in elements], loc=_loc()))


class _DemodTarget:
    """Descriptor produced by demod.full / dual_demod.full / integration.full.
    Records which QUA variables the acquisition writes into."""

    def __init__(self, kind, args, kwargs):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs

    @property
    def targets(self):
        out = [a for a in self.args if isinstance(a, (QuaVar, Index))]
        out += [v for v in self.kwargs.values() if isinstance(v, (QuaVar, Index))]
        return out

    def __repr__(self):
        parts = [a if isinstance(a, str) else repr(a) for a in self.args]
        return f"{self.kind}({', '.join(parts)})"


class _MeasureLib:
    def __init__(self, prefix):
        self._prefix = prefix

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def make(*a, **k):
            return _DemodTarget(f"{self._prefix}.{name}", a, k)
        return make


demod = _MeasureLib("demod")
dual_demod = _MeasureLib("dual_demod")
integration = _MeasureLib("integration")
time_tagging = _MeasureLib("time_tagging")
counting = _MeasureLib("counting")


def measure(pulse, element, *outputs, duration=None, timestamp_stream=None,
            adc_stream=None, stream=None, **_):
    # Legacy signature: measure(pulse, element, adc_stream, *demods)
    outs = list(outputs)
    if outs and (outs[0] is None or isinstance(outs[0], _Stream)):
        adc_stream = adc_stream or outs.pop(0)
    demods = [o for o in outs if isinstance(o, _DemodTarget)]
    targets = [t for d in demods for t in d.targets]
    return _b().emit(Measure(
        _pulse_ref(pulse), _elname(element), stream=adc_stream or stream,
        demods=[repr(d) for d in demods], targets=targets, duration=duration,
        loc=_loc()))


def pause():
    return _b().emit(Pause(loc=_loc()))


def update_frequency(element, new_frequency, units="Hz", keep_phase=False):
    return _b().emit(UpdateFrequency(_elname(element), lit(new_frequency),
                                     units, keep_phase, loc=_loc()))


def update_correction(element, c00, c01, c10, c11):
    return _b().emit(UpdateCorrection(_elname(element),
                                      [lit(c00), lit(c01), lit(c10), lit(c11)],
                                      loc=_loc()))


def frame_rotation(angle, *elements):
    return _b().emit(FrameRotation(lit(angle),
                                   [_elname(e) for e in elements], loc=_loc()))


def frame_rotation_2pi(angle, *elements):
    """Same as frame_rotation but the angle is in units of 2*pi."""
    return _b().emit(FrameRotation(BinOp("*", lit(angle), Lit(2 * math.pi)),
                                   [_elname(e) for e in elements], loc=_loc()))


def reset_frame(*elements):
    return _b().emit(ResetFrame([_elname(e) for e in elements], loc=_loc()))


def reset_phase(element):
    return _b().emit(ResetPhase(_elname(element), loc=_loc()))


reset_if_phase = reset_phase


def set_dc_offset(element, element_input, offset):
    return _b().emit(SetDCOffset(_elname(element), str(element_input),
                                 lit(offset), loc=_loc()))


def ramp_to_zero(element, duration=None):
    return _b().emit(RampToZero(_elname(element), duration, loc=_loc()))


def wait_for_trigger(element, pulse_to_play=None, trigger_element=None,
                     time_tag_target=None, **_):
    return _b().emit(WaitForTrigger(
        _elname(element),
        _pulse_ref(pulse_to_play) if pulse_to_play else None, loc=_loc()))


# ------------------------------------------------------------------ blocks

class _Block:
    """Context manager that collects statements into `node.body`."""

    def __init__(self, node, body=None):
        self.node = node
        self.body = node.body if body is None else body

    def __enter__(self):
        _b().push(self.body)
        return self.node

    def __exit__(self, *exc):
        _b().pop()
        return False


class ArrayLoop:
    """Marker returned by `qualang_tools.loops.from_array`. Sweeping a concrete
    array is exact as a for_each_, so `for_(*from_array(a, arr))` is rewritten
    into one rather than reconstructed from start/stop/step."""

    def __init__(self, values):
        self.values = [v for v in values]


def for_(var, init=None, cond=None, update=None):
    if isinstance(init, ArrayLoop):
        node = ForEach([var], [init.values], loc=_loc())
    else:
        node = For(var, lit(init) if init is not None else None,
                   cond, lit(update) if update is not None else None, loc=_loc())
    _b().emit(node)
    return _Block(node)


def for_each_(var, values):
    """`for_each_(a, [1,2,3])` or `for_each_((a,b), ([1,2],[3,4]))`."""
    vs = list(var) if isinstance(var, (tuple, list)) else [var]
    if isinstance(var, (tuple, list)):
        vals = [list(v) for v in values]
    else:
        vals = [list(values)]
    node = ForEach(vs, vals, loc=_loc())
    _b().emit(node)
    return _Block(node)


def while_(cond):
    node = While(cond, loc=_loc())
    _b().emit(node)
    return _Block(node)


def infinite_loop_():
    node = InfiniteLoop(loc=_loc())
    _b().emit(node)
    return _Block(node)


def if_(cond, unsafe=False, **_):
    node = If(cond, unsafe=unsafe, loc=_loc())
    _b().emit(node)
    return _Block(node)


def _pending_if():
    b = _b()
    node = b.last_if.get(id(b.stack[-1]))
    if node is None:
        raise RuntimeError("elif_/else_ without a preceding if_ block")
    return node


def elif_(cond):
    node = _pending_if()
    body = []
    node.elifs.append((cond, body))
    return _Block(node, body)


def else_():
    node = _pending_if()
    node.orelse = []
    return _Block(node, node.orelse)


_switch_stack: list = []


class _SwitchBlock(_Block):
    """`with switch_(x):` holds no statements itself -- case_/default_ blocks
    attach their bodies to the Switch node while it is on the stack."""

    def __enter__(self):
        _switch_stack.append(self.node)
        return super().__enter__()

    def __exit__(self, *exc):
        r = super().__exit__(*exc)
        _switch_stack.pop()
        return r


def switch_(expression, unsafe=False):
    node = Switch(lit(expression), loc=_loc())
    _b().emit(node)
    return _SwitchBlock(node, [])


def case_(value):
    if not _switch_stack:
        raise RuntimeError("case_ outside switch_")
    body = []
    _switch_stack[-1].cases.append((value, body))
    return _Block(_switch_stack[-1], body)


def default_():
    if not _switch_stack:
        raise RuntimeError("default_ outside switch_")
    _switch_stack[-1].default = []
    return _Block(_switch_stack[-1], _switch_stack[-1].default)


def strict_timing_():
    node = StrictTiming(loc=_loc())
    _b().emit(node)
    return _Block(node)


def port_condition(cond=None):
    node = If(cond if cond is not None else Lit(True), loc=_loc())
    _b().emit(node)
    return _Block(node)


@contextmanager
def stream_processing():
    b = _b()
    b.suppress += 1
    try:
        yield
    finally:
        b.suppress -= 1


# ------------------------------------------------------------------ libraries

class _Lib:
    """`Math.cos(x)` etc. build Call nodes evaluated by the interpreter."""

    def __init__(self, ns):
        self._ns = ns

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args):
            return Call(f"{self._ns}.{name}", [lit(a) for a in args])
        return call


Math = _Lib("Math")
Cast = _Lib("Cast")
Util = _Lib("Util")
Random = _Lib("Random")
broadcast = _Lib("broadcast")      # broadcast.and_/or_/xor_ over run-time values
__all__.append("broadcast")


def __getattr__(name):
    """Anything in the QUA API we have not modelled becomes a no-op statement,
    so an unfamiliar script traces instead of crashing. Reported by the CLI."""
    if name.startswith("__"):
        raise AttributeError(name)

    def unmodelled(*a, **k):
        if _stack:
            _b().emit(Unknown(name, ", ".join(map(str, a))[:80], loc=_loc()))
        return None
    unmodelled.__name__ = name
    unmodelled._qua_unmodelled = True
    return unmodelled


# Parts of the QUA API we record but do not model. They have to be real module
# globals, not just __getattr__ hits: `from qm.qua import *` imports __all__
# only, and a NameError mid-program would abort tracing halfway through.
_UNMODELLED_API = ("reset_global_phase", "assign_variables_to_element",
                   "dual_measure", "wait_for_trigger_all", "set_output_dc_offset")
for _name in _UNMODELLED_API:
    globals()[_name] = __getattr__(_name)
    __all__.append(_name)
