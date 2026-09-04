"""The scheduled result: what each element outputs, when.

A `Timeline` is what the interpreter produces and every renderer consumes. Time
is in nanoseconds from the start of the program; the QUA clock is 4 ns, so all
durations here are multiples of 4 unless a config declares otherwise.

The sampling helpers answer one question -- "what is the voltage range of
element E, channel C, over [ta, tb)?" -- which is all a scope trace needs, and
they answer it exactly (min/max over the underlying samples) so that narrow
pulses cannot vanish between pixels.
"""

import bisect
import math
from dataclasses import dataclass, field
from typing import Optional

CLOCK_NS = 4        # one QUA clock cycle


@dataclass
class Segment:
    """A pulse occupying an element for [t0, t1)."""
    element: str
    kind: str                    # 'play' | 'measure' | 'ramp' | 'trigger'
    t0: float
    t1: float
    op: str = ""                 # operation name as written in the script
    pulse: str = ""              # resolved pulse name
    waves: dict = field(default_factory=dict)    # channel -> Waveform
    amp: float = 1.0             # amp() scaling actually applied
    amp_label: str = ""          # e.g. "*a" when the scale is a QUA variable
    freq: float = 0.0            # IF in Hz at play time
    phase: float = 0.0           # frame phase in rad at play time
    runtime: bool = False        # inside a branch chosen from unknown data
    loc: str = ""
    iteration: tuple = ()
    note: str = ""

    @property
    def duration(self):
        return self.t1 - self.t0


@dataclass
class Acquisition:
    """A measurement's ADC window (time_of_flight and smearing applied)."""
    element: str
    t0: float
    t1: float
    label: str = ""
    loc: str = ""


@dataclass
class Marker:
    """A zero-duration event: align, pause, frame rotation, frequency update."""
    kind: str
    t: float
    elements: list = field(default_factory=list)
    label: str = ""
    loc: str = ""


@dataclass
class Break:
    """Where loop unrolling was cut short."""
    t: float
    skipped: Optional[int]        # iterations not drawn (None = unknown)
    label: str = ""


@dataclass
class LevelChange:
    """A step in the DC level of one channel (set_dc_offset, sticky hold)."""
    element: str
    channel: str
    t: float
    level: float
    reason: str = ""


class Timeline:
    def __init__(self, machine=None):
        self.machine = machine
        self.segments = []
        self.acquisitions = []
        self.markers = []
        self.breaks = []
        self.levels = []
        self.elements = []            # display order (first use)
        self.cursors = {}             # element -> final cursor (ns)
        self.warnings = []
        self.stats = {}
        self.duration = 0.0
        self._index = None

    # ---------------------------------------------------------------- build
    def touch(self, element):
        if element not in self.elements:
            self.elements.append(element)

    def add_segment(self, seg):
        self.touch(seg.element)
        self.segments.append(seg)
        self.duration = max(self.duration, seg.t1)
        self._index = None

    def add(self, rec):
        if isinstance(rec, Segment):
            return self.add_segment(rec)
        if isinstance(rec, Acquisition):
            self.acquisitions.append(rec)
            self.touch(rec.element)
        elif isinstance(rec, Marker):
            self.markers.append(rec)
        elif isinstance(rec, Break):
            self.breaks.append(rec)
        elif isinstance(rec, LevelChange):
            self.levels.append(rec)
            self.touch(rec.element)
        self.duration = max(self.duration, getattr(rec, "t1", None)
                            or getattr(rec, "t", 0.0))

    # ---------------------------------------------------------------- query
    def _build_index(self):
        """Per-(element, channel) segment and level lists, sorted, with the key
        arrays a bisect needs: the renderer asks for one pixel column at a time
        and must not rescan the whole sequence each call."""
        idx = {}
        for s in self.segments:
            for ch in s.waves:
                idx.setdefault((s.element, ch), []).append(s)
        segs = {}
        for k, v in idx.items():
            v.sort(key=lambda s: s.t0)
            segs[k] = (v, [s.t1 for s in v])      # t1 sorted: no overlap per element
        lv = {}
        for L in sorted(self.levels, key=lambda x: x.t):
            lv.setdefault((L.element, L.channel), []).append(L)
        levels = {k: (v, [x.t for x in v]) for k, v in lv.items()}
        self._index = (segs, levels)

    def channels(self, element):
        """Channels this element actually drove, in config order."""
        el = self.machine.element(element) if self.machine else None
        order = el.channels if el else []
        seen = []
        for s in self.segments:
            if s.element == element:
                for ch in s.waves:
                    if ch not in seen:
                        seen.append(ch)
        for L in self.levels:
            if L.element == element and L.channel not in seen:
                seen.append(L.channel)
        return [c for c in order if c in seen] + [c for c in seen if c not in order]

    def level_at(self, element, channel, t):
        if self._index is None:
            self._build_index()
        entry = self._index[1].get((element, channel))
        if not entry:
            return 0.0
        recs, ts = entry
        i = bisect.bisect_right(ts, t)
        return recs[i - 1].level if i else 0.0

    def range_over(self, element, channel, ta, tb, modulate=False):
        """(min, max) volts on a channel over [ta, tb).

        `modulate` mixes the envelope with the element's IF and frame phase --
        what the port actually emits; the default shows the envelope, which is
        what you want to read timing off.
        """
        if self._index is None:
            self._build_index()
        lo = hi = self.level_at(element, channel, ta)
        entry = self._index[0].get((element, channel))
        if not entry:
            return lo, hi
        segs, t1s = entry
        for s in segs[bisect.bisect_right(t1s, ta):]:
            if s.t0 >= tb:
                break
            wf = s.waves[channel]
            mm = wf.minmax(ta - s.t0, tb - s.t0, s.t1 - s.t0)
            if mm is None:
                continue
            a, b = mm[0] * s.amp, mm[1] * s.amp
            base = self.level_at(element, channel, max(ta, s.t0))
            if modulate and s.freq:
                span = (min(tb, s.t1) - max(ta, s.t0)) * 1e-9
                env = max(abs(a), abs(b))
                if span * abs(s.freq) >= 0.5:            # column spans a period
                    a, b = -env, env
                else:
                    ph0 = 2 * math.pi * s.freq * (max(ta, s.t0) - s.t0) * 1e-9 + s.phase
                    ph1 = 2 * math.pi * s.freq * (min(tb, s.t1) - s.t0) * 1e-9 + s.phase
                    c = [math.cos(ph0), math.cos(ph1)]
                    a, b = env * min(c), env * max(c)
            lo = min(lo, base + a, base + b)
            hi = max(hi, base + a, base + b)
        return lo, hi

    def peak(self, element, channel):
        """Largest |V| the channel reaches, held DC level included.

        The level has to be added inside the abs, not outside: a ramp_to_zero
        sits on top of a +60 mV hold and ends at 0, so |level| + |waveform|
        would claim 120 mV and squash the whole lane.
        """
        p = abs(self.level_at(element, channel, self.duration))
        for L in self.levels:
            if L.element == element and L.channel == channel:
                p = max(p, abs(L.level))
        for s in self.segments:
            if s.element != element or channel not in s.waves:
                continue
            wf = s.waves[channel]
            mm = wf.minmax(0.0, s.t1 - s.t0, s.t1 - s.t0)
            if mm is None:
                continue
            base = self.level_at(element, channel, s.t0)
            p = max(p, abs(base + mm[0] * s.amp), abs(base + mm[1] * s.amp))
        return p

    # ---------------------------------------------------------------- misc
    def summary(self):
        n_play = sum(1 for s in self.segments if s.kind == "play")
        return (f"{len(self.elements)} elements, {n_play} pulses, "
                f"{len(self.acquisitions)} measurements, "
                f"{fmt_time(self.duration)} total")


def fmt_time(ns):
    if ns >= 1e9:
        return f"{ns / 1e9:.4g} s"
    if ns >= 1e6:
        return f"{ns / 1e6:.4g} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.4g} us"
    return f"{ns:.4g} ns"


def fmt_freq(hz):
    a = abs(hz)
    if a >= 1e9:
        return f"{hz / 1e9:.4g} GHz"
    if a >= 1e6:
        return f"{hz / 1e6:.4g} MHz"
    if a >= 1e3:
        return f"{hz / 1e3:.4g} kHz"
    return f"{hz:.4g} Hz"
