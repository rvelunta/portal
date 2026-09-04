"""Run a user's QUA script with the SDK faked out, and capture what matters.

`load_script()` installs stand-in modules (`qm`, `qm.qua`, `qualang_tools`,
matplotlib, and numpy only if the real one is missing), then execs the script
as `__main__` with its own directory on sys.path so `from configuration import
config` works.

The script is expected to fail or no-op at the point where it would talk to
hardware -- that is fine. Programs are captured as each `with program()` block
CLOSES, so a later `qmm.open_qm(...)` blowing up still leaves us the sequence.
A wall-clock alarm bounds scripts that poll a fetching loop.
"""

import os
import signal
import sys
import types

from . import qua as qua_mod
from .ir import Program

__all__ = ["load_script", "LoadResult", "install_stubs"]


class LoadResult:
    def __init__(self, programs, configs, warnings, globals_):
        self.programs = programs        # [ir.Program]
        self.configs = configs          # [dict] -- QM config dicts found
        self.warnings = warnings
        self.globals = globals_

    @property
    def program(self):
        if not self.programs:
            raise RuntimeError("no `with program()` block was traced")
        return self.programs[-1]

    @property
    def config(self):
        if not self.configs:
            raise RuntimeError(
                "no QM configuration dict found -- pass --config module:name"
            )
        return self.configs[0]


# ------------------------------------------------------------------ stubs

class _Any:
    """Permissive stand-in for hardware handles (job, result handles, octave).

    Deliberately falsy and empty-iterable: scripts commonly spin on
    `while job.result_handles.is_processing():`, which must terminate here.
    """

    def __init__(self, name="qm"):
        self.__dict__["_name"] = name

    def __call__(self, *a, **k): return _Any(f"{self._name}()")
    def __getattr__(self, item): return _Any(f"{self._name}.{item}")
    def __setattr__(self, k, v): self.__dict__[k] = v
    def __getitem__(self, k): return _Any(f"{self._name}[{k}]")
    def __iter__(self): return iter(())
    def __len__(self): return 0
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __repr__(self): return f"<stub {self._name}>"
    def __float__(self): return 0.0
    def __int__(self): return 0
    def __add__(self, o): return self
    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    for k, v in attrs.items():
        if isinstance(v, types.ModuleType):
            sys.modules[f"{name}.{k}"] = v
    sys.modules[name] = m
    return m


class _StubModule(types.ModuleType):
    """Module whose every attribute is a permissive stub, and whose every
    submodule import also succeeds."""

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        sub = _StubModule(f"{self.__name__}.{item}")
        sys.modules[sub.__name__] = sub
        setattr(self, item, sub)
        return sub

    def __call__(self, *a, **k):
        return _Any(self.__name__)


def _stub_package(name, warnings=None):
    if warnings is not None:
        warnings.append(f"stubbed import: {name}")
    m = _StubModule(name)
    m.__path__ = []
    sys.modules[name] = m
    return m


class _unit:
    """qualang_tools.units.unit(): multipliers used all over QM configs."""

    def __init__(self, coerce_to_integer=False):
        self._int = coerce_to_integer
        vals = dict(ns=1, us=1000, ms=int(1e6), s=int(1e9), clock_cycle=4,
                    Hz=1, kHz=1e3, MHz=1e6, GHz=1e9, uHz=1e-6, mHz=1e-3,
                    V=1, mV=1e-3, uV=1e-6, A=1, mA=1e-3, uA=1e-6, nA=1e-9)
        for k, v in vals.items():
            setattr(self, k, int(v) if coerce_to_integer and v >= 1 else v)

    def to_clock_cycles(self, t): return int(t // 4)
    def demod2volts(self, data, duration, **k): return data


def _make_qm_package(captured_configs, warnings):
    """The `qm` package: real tracing DSL, stubbed hardware."""

    class QuantumMachinesManager:
        def __init__(self, *a, **k):
            self.args = (a, k)

        def open_qm(self, config=None, *a, **k):
            if isinstance(config, dict):
                captured_configs.append(config)
            return _QuantumMachine()

        open_qm_with_config = open_qm

        def simulate(self, config=None, program=None, *a, **k):
            if isinstance(config, dict):
                captured_configs.append(config)
            return _Any("job")

        def __getattr__(self, item): return _Any(f"qmm.{item}")

    class _QuantumMachine:
        def execute(self, prog=None, *a, **k): return _Any("job")
        def simulate(self, *a, **k): return _Any("job")
        def compile(self, *a, **k): return _Any("pid")
        def close(self, *a, **k): return None
        def __getattr__(self, item): return _Any(f"qm.{item}")

    def generate_qua_script(prog=None, config=None):
        if isinstance(config, dict):
            captured_configs.append(config)
        from .textdump import program_to_text
        return program_to_text(prog) if isinstance(prog, Program) else ""

    qm = _module(
        "qm",
        qua=qua_mod,
        QuantumMachinesManager=QuantumMachinesManager,
        QuantumMachine=_QuantumMachine,
        SimulationConfig=lambda *a, **k: _Any("SimulationConfig"),
        LoopbackInterface=lambda *a, **k: _Any("LoopbackInterface"),
        ControllerConnection=lambda *a, **k: _Any("ControllerConnection"),
        InterOpxChannel=lambda *a, **k: _Any("InterOpxChannel"),
        generate_qua_script=generate_qua_script,
        Program=Program,
    )
    sys.modules["qm.qua"] = qua_mod
    lib = types.ModuleType("qm.qua.lib")       # `from qm.qua.lib import Math`
    for n in ("Math", "Cast", "Util", "Random", "broadcast"):
        setattr(lib, n, getattr(qua_mod, n))
    sys.modules["qm.qua.lib"] = lib
    qua_mod.lib = lib
    # Old import paths: `from qm.QuantumMachinesManager import QuantumMachinesManager`
    for legacy, attr in (("qm.QuantumMachinesManager", QuantumMachinesManager),
                         ("qm.QuantumMachine", _QuantumMachine),
                         ("qm.program", qua_mod),
                         ("qm.simulate", None)):
        m = types.ModuleType(legacy)
        if attr is not None:
            setattr(m, legacy.split(".")[-1], attr)
        m.__getattr__ = lambda item, _l=legacy: _Any(f"{_l}.{item}")
        sys.modules[legacy] = m
    for extra in ("qm.octave", "qm.simulate.credentials", "qm.jobs",
                  "qm.results", "qm.api", "qm.exceptions", "qm.logger",
                  "qm.type_hinting", "qm.grpc", "qm.utils", "qm.serialization"):
        _stub_package(extra)
    return qm


def _make_qualang_tools(warnings):
    def from_array(var, array):
        """Exact rewrite: sweep the concrete array with for_each_ semantics."""
        return (var, qua_mod.ArrayLoop([_num(v) for v in array]), None, None)

    def qua_arange(var, start, stop, step):
        vals, v = [], start
        while (step > 0 and v < stop) or (step < 0 and v > stop):
            vals.append(v)
            v += step
        return (var, qua_mod.ArrayLoop(vals), None, None)

    loops = _module("qualang_tools.loops", from_array=from_array,
                    qua_arange=qua_arange)
    units = _module("qualang_tools.units", unit=_unit)
    results = _stub_package("qualang_tools.results")
    plot = _stub_package("qualang_tools.plot")
    for name in ("qualang_tools.bakery", "qualang_tools.config",
                 "qualang_tools.addons", "qualang_tools.analysis",
                 "qualang_tools.characterization", "qualang_tools.simulator",
                 "qualang_tools.voltage_gates", "qualang_tools.multi_user",
                 "qualang_tools.octave_tools", "qualang_tools.digital_filters"):
        _stub_package(name)
    pkg = _module("qualang_tools", loops=loops, units=units, results=results,
                  plot=plot)
    pkg.__path__ = []
    pkg.__getattr__ = lambda item: _stub_package(f"qualang_tools.{item}")
    return pkg


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def install_stubs(captured_configs, warnings):
    """Put the stand-in modules into sys.modules. Returns a restore callable."""
    saved = {k: sys.modules.get(k) for k in list(sys.modules)
             if k == "qm" or k.startswith("qm.") or k.startswith("qualang_tools")}

    _make_qm_package(captured_configs, warnings)
    _make_qualang_tools(warnings)

    try:
        import numpy  # noqa: F401
    except ImportError:
        from . import _numpy_shim
        sys.modules["numpy"] = _numpy_shim
        sys.modules["numpy.random"] = _numpy_shim.random
        warnings.append(
            "numpy not installed: using qua_scope's minimal numpy stand-in")

    finder = _StubFinder(warnings)
    sys.meta_path.append(finder)

    def restore():
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return restore


class _StubFinder:
    """Last-resort import hook: anything the script imports that is not
    installed (matplotlib, scipy, a lab package) becomes a permissive stub, so
    the sequence still traces. Every stub is reported.

    It sits at the END of sys.meta_path, so real modules -- including the
    script's own `configuration.py` -- always win.
    """

    NEVER = ("qm", "qm.qua", "qua_scope")

    def __init__(self, warnings):
        self.warnings = warnings

    def find_module(self, fullname, path=None):        # legacy API, unused
        return None

    def find_spec(self, fullname, path=None, target=None):
        import importlib.machinery
        if fullname in self.NEVER or fullname.startswith("qua_scope"):
            return None
        return importlib.machinery.ModuleSpec(fullname, _StubLoader(self))


class _StubLoader:
    def __init__(self, finder):
        self.finder = finder

    def create_module(self, spec):
        top = spec.name.split(".")[0]
        if spec.name == top:
            self.finder.warnings.append(f"stubbed missing import: {spec.name}")
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        return None


# ------------------------------------------------------------------ execution

class _Timeout(Exception):
    pass


def _alarm(seconds):
    """Bound script wall time where the platform allows it."""
    try:
        def handler(signum, frame):
            raise _Timeout(f"script exceeded {seconds}s")
        old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)

        def cancel():
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        return cancel
    except (ValueError, AttributeError):
        return lambda: None


def _looks_like_config(obj):
    return (isinstance(obj, dict) and "elements" in obj
            and ("pulses" in obj or "waveforms" in obj))


def load_script(path, timeout=30.0, extra_config=None):
    """Exec a QUA script under the stand-in SDK; return a LoadResult."""
    path = os.path.abspath(path)
    warnings = []
    captured_configs = []
    restore = install_stubs(captured_configs, warnings)
    qua_mod.reset_trace()

    script_dir = os.path.dirname(path)
    sys.path.insert(0, script_dir)
    g = {"__name__": "__main__", "__file__": path, "__builtins__": __builtins__}
    cancel = _alarm(timeout)
    try:
        with open(path) as f:
            src = f.read()
        try:
            exec(compile(src, path, "exec"), g)
        except _Timeout as e:
            warnings.append(f"{e} -- stopped early; traced program kept")
        except SystemExit:
            pass
        except Exception as e:                     # hardware calls, plotting...
            warnings.append(
                f"script raised after tracing: {type(e).__name__}: {e}")
    finally:
        cancel()
        if sys.path and sys.path[0] == script_dir:
            sys.path.pop(0)
        restore()

    programs = list(qua_mod.traced_programs)
    configs = list(captured_configs)
    for name, val in g.items():                     # module-level `config = {...}`
        if _looks_like_config(val) and not any(val is c for c in configs):
            configs.append(val)
    if extra_config is not None:
        configs.insert(0, extra_config)
    if not configs:
        for mod in list(sys.modules.values()):      # e.g. `configuration.py`
            try:
                d = getattr(mod, "config", None)
            except Exception:
                continue
            if _looks_like_config(d):
                configs.append(d)
                break

    unmodelled = sorted({s.name for p in programs for s in _walk(p.body)
                         if type(s).__name__ == "Unknown"})
    if unmodelled:
        warnings.append("QUA statements recorded but not modelled: "
                        + ", ".join(unmodelled))
    return LoadResult(programs, configs, warnings, g)


def _walk(body):
    for s in body:
        yield s
        for attr in ("body", "orelse", "default"):
            sub = getattr(s, attr, None)
            if sub:
                yield from _walk(sub)
        for _, sub in getattr(s, "elifs", []) or []:
            yield from _walk(sub)
        for _, sub in getattr(s, "cases", []) or []:
            yield from _walk(sub)
