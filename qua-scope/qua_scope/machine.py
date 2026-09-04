"""The QM configuration dict, resolved into something a renderer can sample.

Only the parts that affect what a scope would show are modelled: elements and
their ports/IF, pulses and their lengths, waveforms, digital markers, and the
measurement fields (time_of_flight, smearing). Mixers, correction matrices and
controller-level settings are read but not applied.

Waveforms stay lazy -- a 10 us constant readout pulse is ('const', 0.1), not
10000 samples -- because the renderer only ever asks for values at pixel
resolution.
"""

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_TOF = 24          # ns; QM's minimum time_of_flight when unspecified


@dataclass
class Waveform:
    name: str
    kind: str                     # 'const' | 'arb'
    value: float = 0.0            # const
    samples: list = field(default_factory=list)   # arb, 1 sample per ns

    def at(self, t_ns, length):
        """Value at t_ns into a pulse of `length` ns (0 outside)."""
        if t_ns < 0 or t_ns >= length:
            return 0.0
        if self.kind == "const":
            return self.value
        n = len(self.samples)
        if n == 0:
            return 0.0
        i = int(t_ns * n / length) if length != n else int(t_ns)
        return self.samples[min(max(i, 0), n - 1)]

    def minmax(self, ta, tb, length):
        """Exact (min, max) of the envelope over [ta, tb) ns into the pulse.

        Exact rather than point-sampled: at pixel resolution a 40 ns Gaussian
        can fall inside one column, and a point sample would miss its peak.
        """
        ta, tb = max(ta, 0.0), min(tb, float(length))
        if tb <= ta:
            return None
        if self.kind == "const":
            return (self.value, self.value)
        n = len(self.samples)
        if n == 0 or length <= 0:
            return (0.0, 0.0)
        i0 = int(ta * n / length)
        i1 = max(i0 + 1, int(-(-tb * n // length)))   # ceil
        chunk = self.samples[max(0, i0):min(n, i1)] or [0.0]
        return (min(chunk), max(chunk))

    @property
    def peak(self):
        if self.kind == "const":
            return abs(self.value)
        return max((abs(s) for s in self.samples), default=0.0)


ZERO_WF = Waveform("zero", "const", 0.0)


@dataclass
class Pulse:
    name: str
    length: int                   # ns
    operation: str = "control"    # 'control' | 'measurement'
    waveforms: dict = field(default_factory=dict)   # 'single'|'I'|'Q' -> wf name
    digital_marker: Optional[str] = None
    integration_weights: dict = field(default_factory=dict)


@dataclass
class Element:
    name: str
    kind: str = "unknown"         # 'mix' | 'single' | 'multi' | 'unknown'
    ports: dict = field(default_factory=dict)       # 'I'/'Q'/'single' -> (con, n)
    intermediate_frequency: float = 0.0
    lo_frequency: float = 0.0
    operations: dict = field(default_factory=dict)  # op name -> pulse name
    time_of_flight: int = DEFAULT_TOF
    smearing: int = 0
    outputs: dict = field(default_factory=dict)
    digital_inputs: dict = field(default_factory=dict)
    sticky: bool = False
    sticky_duration: int = 0
    thread: Optional[str] = None

    @property
    def channels(self):
        """Analog channels this element drives, in display order."""
        if self.kind == "mix":
            return ["I", "Q"]
        if self.kind == "single":
            return ["single"]
        return list(self.ports.keys()) or ["single"]

    @property
    def port_label(self):
        parts = [f"{k}:{v[0]}/{v[1]}" for k, v in self.ports.items()
                 if isinstance(v, (tuple, list)) and len(v) >= 2]
        return " ".join(parts)


class Machine:
    """Resolved view of a QM config dict."""

    def __init__(self, config):
        self.config = config or {}
        self.waveforms = {}
        self.pulses = {}
        self.elements = {}
        self.digital_waveforms = dict(self.config.get("digital_waveforms", {}))
        self.warnings = []
        self._load()

    # ---------------------------------------------------------------- loading
    def _load(self):
        for name, wf in (self.config.get("waveforms") or {}).items():
            t = wf.get("type", "constant")
            if t == "constant":
                self.waveforms[name] = Waveform(name, "const",
                                                float(wf.get("sample", 0.0)))
            else:
                samples = [float(s) for s in _as_list(wf.get("samples", []))]
                self.waveforms[name] = Waveform(name, "arb", samples=samples)

        for name, p in (self.config.get("pulses") or {}).items():
            self.pulses[name] = Pulse(
                name=name,
                length=int(p.get("length", 0)),
                operation=p.get("operation", "control"),
                waveforms=dict(p.get("waveforms", {}) or {}),
                digital_marker=p.get("digital_marker"),
                integration_weights=dict(p.get("integration_weights", {}) or {}),
            )

        for name, e in (self.config.get("elements") or {}).items():
            el = Element(name=name)
            if "mixInputs" in e:
                el.kind = "mix"
                mi = e["mixInputs"]
                for k in ("I", "Q"):
                    if k in mi:
                        el.ports[k] = tuple(mi[k])
                el.lo_frequency = float(mi.get("lo_frequency", 0) or 0)
            elif "singleInput" in e:
                el.kind = "single"
                el.ports["single"] = tuple(e["singleInput"].get("port", ()))
            elif "MWInput" in e:                       # OPX1000 MW-FEM element
                el.kind = "single"
                el.ports["single"] = tuple(e["MWInput"].get("port", ()))
                el.lo_frequency = float(e["MWInput"].get("upconverter", 0) or 0)
            elif "multipleInputs" in e:
                el.kind = "multi"
                for k, v in (e["multipleInputs"].get("inputs") or {}).items():
                    el.ports[k] = tuple(v)
            elif "singleInputCollection" in e:
                el.kind = "multi"
                for k, v in (e["singleInputCollection"].get("inputs") or {}).items():
                    el.ports[k] = tuple(v)

            el.intermediate_frequency = float(
                e.get("intermediate_frequency", 0) or 0)
            el.operations = dict(e.get("operations", {}) or {})
            el.time_of_flight = int(e.get("time_of_flight", DEFAULT_TOF) or 0)
            el.smearing = int(e.get("smearing", 0) or 0)
            el.outputs = dict(e.get("outputs", {}) or {})
            el.digital_inputs = dict(e.get("digitalInputs", {}) or {})
            el.thread = e.get("thread")
            hold = e.get("hold_offset") or (e.get("sticky") or {})
            if hold:
                el.sticky = bool(hold.get("analog", True))
                el.sticky_duration = int(hold.get("duration", 0) or 0)
            self.elements[name] = el

    # ---------------------------------------------------------------- lookups
    def element(self, name):
        el = self.elements.get(name)
        if el is None:
            el = Element(name=name)
            self.elements[name] = el
            self.warnings.append(f"element '{name}' is not in the config")
        return el

    def pulse_for(self, element, operation):
        """Resolve `play(op, element)` to a Pulse via the element's op map."""
        el = self.element(element)
        pname = el.operations.get(operation, operation)
        p = self.pulses.get(pname)
        if p is None:
            self.warnings.append(
                f"operation '{operation}' on '{element}' has no pulse in the "
                f"config (assuming 16 ns)")
            p = Pulse(name=str(pname), length=16)
            self.pulses[str(pname)] = p
        return p

    def waveform(self, name):
        wf = self.waveforms.get(name)
        if wf is None and name is not None:
            self.warnings.append(f"waveform '{name}' is not in the config")
        return wf or ZERO_WF


def _as_list(x):
    """Config samples may be a numpy array, a list, or a list of (v, n) pairs
    (QM's compressed arbitrary-waveform form)."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    for s in x:
        if isinstance(s, (tuple, list)) and len(s) == 2:
            out.extend([float(s[0])] * int(s[1]))     # compressed run-length
        else:
            out.append(float(s))
    return out
