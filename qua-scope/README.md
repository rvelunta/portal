# qua-scope

Give it a Quantum Machines QUA script, get the pulse sequence it would play.
Pure stdlib -- no `qm` SDK, no numpy, no hardware.

```
python3 -m qua_scope examples/power_rabi.py            # -> power_rabi.scope.svg
python3 -m qua_scope examples/ramsey.py --text         # ASCII scope in the terminal
python3 -m qua_scope my_script.py --dump               # the traced QUA program
python3 -m qua_scope my_script.py -i 4 --modulate      # 4 loop iterations, IF shown
```

The script itself is unmodified QUA. `qm`, `qm.qua` and `qualang_tools` are
replaced by tracing stand-ins, so `QuantumMachinesManager(...)`, `open_qm`,
`execute` and the plotting tail all no-op while the sequence is captured.

## How it works
1. **`loader.py`** execs the script as `__main__` with the SDK faked out and its
   own directory on `sys.path` (so `from configuration import config` works).
   Programs are captured as each `with program()` block closes, so a later
   hardware call blowing up still leaves the sequence intact. Imports that are
   not installed (matplotlib, scipy, a lab package) become permissive stubs and
   are reported.
2. **`qua.py`** is the tracing DSL. `play`/`wait`/`align`/... append IR nodes;
   `with for_(...)`/`if_(...)` open nested blocks. Because `n < 100` has no
   value at trace time, QUA variables build expression trees instead.
3. **`interp.py`** unrolls the tree with concrete variable values and schedules
   it: one cursor per element, `align` = max, `wait` in clock cycles.
4. **`render_svg.py` / `render_text.py`** draw the resulting `Timeline`.

## What is modelled
- Per-element timelines. A statement advances only the elements it names;
  `align(a, b)` snaps them to the latest cursor, `align()` covers every element
  the program used.
- `wait(n, el)` = n clock cycles = 4n ns. Pulse length comes from the config;
  `duration=` and `truncate=` (clock cycles) override it.
- `play("op" * amp(a), el)` -- the amplitude scale is evaluated per iteration,
  so a swept `a` really is 0, 0.25, 0.5 ... in the picture and in the labels.
- `measure(...)`: the element is busy for the pulse length, and the ADC window
  is drawn from `time_of_flight - smearing` to `length + time_of_flight +
  smearing`.
- Sticky elements (`hold_offset`): each pulse adds to the held level and the
  level persists across idle time; `ramp_to_zero` brings it back. This is what
  makes a charge-stability raster look like a staircase.
- `set_dc_offset`, `update_frequency`, `frame_rotation(_2pi)`, `reset_frame`,
  `reset_phase`, `wait_for_trigger`, `pause`, `ramp(rate)`, `strict_timing_`.
- Control flow: `for_`, `for_each_`, `while_`, `if_/elif_/else_`, `switch_`,
  `infinite_loop_`, and `qualang_tools.loops.from_array` (rewritten to an exact
  `for_each_` over the array).

## Loops and run-time values -- read this before trusting a picture
- **Loops are truncated.** Each loop draws `--iterations` passes (default 2) and
  a hatched break marked `x N more`. The count is exact when the body does not
  touch the loop variable; otherwise it shows `?`. `--full` unrolls everything,
  bounded by `--max-segments`.
- **Values that only exist at run time are unknown here**: demodulation results,
  input streams, `Random`. They evaluate to 0, and any branch taken on one is
  marked "run-time branch" in the diagram (purple tint on the pulses inside it).
  `--branch then|else` forces the other side. An active-reset loop therefore
  shows one plausible execution, not the only one.
- **The axis is not linear by default.** Idle stretches longer than a couple of
  pixels are collapsed to a zigzag band labelled with the time skipped, because
  otherwise a 40 ns pi pulse next to a 10 us wait is a quarter of a pixel.
  `--no-compress` gives a strictly linear axis.

## Deliberate omissions
- Ideal timing: no pulse-processor overhead between statements, no analog
  output filter delay, no OPX real-time latency. Two statements 16 ns apart in
  the model can be further apart on hardware.
- `fixed` is evaluated in float, not 4.28 fixed-point (except where a QUA op is
  integer-truncating, e.g. `Cast.mul_int_by_fixed`).
- Mixer correction matrices, LO leakage and IQ imbalance are read but not
  applied -- `--modulate` mixes envelope x cos(2 pi IF t + frame) only.
- Digital markers, integration weights and stream processing are parsed but not
  drawn; only the ADC window is.
- `measure` blocks its element for the pulse length; demodulation/processing
  time is not added.

## Files
- `qua_scope/ir.py` -- statement and expression nodes, with source locations.
- `qua_scope/qua.py` -- the `qm.qua` stand-in (tracing).
- `qua_scope/loader.py` -- runs the script, fakes the SDK, finds the config.
- `qua_scope/machine.py` -- the QM config resolved into elements/pulses/waveforms.
- `qua_scope/interp.py` -- unrolling and the timing model.
- `qua_scope/timeline.py` -- the scheduled result and its sampling helpers.
- `qua_scope/axis.py` -- the (optionally gap-collapsing) time axis.
- `qua_scope/render_svg.py`, `render_text.py`, `textdump.py` -- output.
- `qua_scope/_numpy_shim.py` -- minimal numpy, used only when numpy is missing.
- `examples/` -- power Rabi, Ramsey, active reset, and a sticky-gate charge
  stability sweep (config inline).
- `tests/test_timing.py` -- the contract. `python3 tests/test_timing.py`.

## API
```python
from qua_scope import load_script, Machine, Interpreter, RunConfig, render_svg

res = load_script("my_script.py")          # res.programs, res.configs, res.warnings
tl = Interpreter(Machine(res.config), RunConfig(max_iterations=4)).run(res.program)
open("out.svg", "w").write(render_svg(tl, title="my sequence"))

for s in tl.segments[:5]:
    print(s.element, s.op, s.t0, s.t1, s.amp)
```

`--json out.json` writes the same timeline for other tools.
