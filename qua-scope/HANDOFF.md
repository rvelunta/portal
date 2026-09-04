# qua-scope -- handoff notes for another Claude instance

You are picking up `qua-scope`: a tool that takes an unmodified Quantum Machines
QUA script (`.py`) and draws the pulse sequence it would play, as an SVG or an
ASCII scope. Written 2026-09-04. Pure Python stdlib -- **this machine has no
pip, no numpy, no `qm` SDK**, so do not reach for dependencies; if you need
array maths, extend `_numpy_shim.py`.

Read `README.md` for the user-facing description. This file is the parts you
need to work on it: why it is shaped this way, the invariants, and the traps.

## The user's task

They hand you a QUA script (often with a `configuration.py` next to it) and
want a scope diagram back. Do this:

```bash
/home/roland/projects/portal/qua-scope/qua-scope <script.py>          # -> <script>.scope.svg
/home/roland/projects/portal/qua-scope/qua-scope <script.py> --text   # ASCII, so YOU can see it
```

The wrapper works from any cwd. `python3 -m qua_scope <script>` also works from
the `qua-scope/` directory. **Always run `--text` yourself** before reporting:
you cannot see the SVG, but the ASCII scope is the same Timeline through the
same axis, so it is how you check the answer is sane (element order, pulse
widths, gaps, markers). `--dump` prints the traced program if the picture looks
wrong and you need to know whether tracing or scheduling is at fault.

Useful flags: `-i N` / `--full` (loop iterations), `--branch then|else`,
`--modulate`, `--no-compress`, `--elements a,b`, `--tmin/--tmax 2us`,
`--config module:name`, `--json out.json`, `--theme dark`, `--program -2`.

## Pipeline (four stages, each a module you can debug alone)

```
script.py --loader.py--> ir.Program --interp.py--> Timeline --render_*.py--> SVG / text
             (exec with          (tree with        (flat, scheduled,
              a fake SDK)         loops)            per element)
```

1. **`loader.py`** -- execs the script as `__main__` with `qm`, `qm.qua`,
   `qm.qua.lib`, `qualang_tools.{loops,units}` replaced by stand-ins, the
   script's directory on `sys.path`, a `SIGALRM` wall-clock bound, and a
   last-resort `sys.meta_path` finder that stubs any import that is not
   installed (reported as a warning). Programs are captured when each
   `with program()` block **closes**, so a later hardware call raising still
   leaves the sequence. Config is found from `open_qm(cfg)`/`simulate(cfg,...)`,
   then from script globals, then from any imported module's `config`.
2. **`qua.py`** -- the tracing `qm.qua`. Statements append IR nodes; `with
   for_/if_/switch_` push nested blocks. QUA variables are `ir.QuaVar` with
   overloaded operators, so `n < 100` builds a `BinOp`, not a bool.
3. **`interp.py`** -- unrolls the tree with concrete values and schedules it.
   This is where the timing model lives.
4. **`render_svg.py` / `render_text.py`** -- draw a `Timeline` through
   `axis.py`'s piecewise time->pixel map.

Supporting: `ir.py` (nodes + `Loc`), `machine.py` (QM config resolved into
elements/pulses/waveforms, lazy sampling), `timeline.py` (the scheduled result
and its exact min/max sampling), `textdump.py` (`--dump`), `cli.py`,
`_numpy_shim.py`.

## The timing model (interp.py) -- the thing to get right

- One cursor per element (ns). A statement advances **only** the elements it
  names. This is QUA's actual semantics: elements are independent threads.
- `align(a, b)` = set both cursors to `max`. Bare `align()` covers **every
  element the program mentions anywhere**, including ones first played later --
  QUA compiles the whole program, so `interp._elements_in()` pre-scans the tree.
  (This was a real bug: originally it only covered elements already touched.)
- `wait(n, el)` = `4n` ns. `play(..., duration=d)` = `4d` ns and overrides the
  config length; `truncate=t` shortens to `4t`.
- `measure` = a play of the measurement pulse (element busy for the pulse
  length only, no demod/processing time) plus an `Acquisition` from
  `t0 + time_of_flight - smearing` to `t1 + time_of_flight + smearing`.
- Sticky elements (`hold_offset`/`sticky` in the element config): each pulse
  **adds** its last sample to a held level that persists across idle time;
  `ramp_to_zero` ramps it back. `set_dc_offset` sets the level outright. Levels
  are `LevelChange` records, sampled by `Timeline.level_at`.
- `update_frequency`, `frame_rotation(_2pi)`, `reset_frame/phase` are recorded
  on the segment (`freq`, `phase`) and cost no time.
- Ideal timing: no pulse-processor overhead between statements, no analog
  filter delay. Say so when reporting; two statements 16 ns apart here can be
  further apart on hardware.

### Unrolling and unknown values

- Loops draw `RunConfig.max_iterations` passes (default 2) then emit a `Break`
  with `skipped` = the exact remaining count when the body does not assign the
  loop variable (`_assigns` guards this), else `None` -> drawn as `?`.
- Anything only known at run time -- demod results, input streams, `Random` --
  is **tainted**: `_ev()` returns `(value, tainted)`, tainted values evaluate to
  0, and taint propagates through `assign`. A branch taken on a tainted
  condition emits a `branch` marker, sets `runtime=True` on the segments
  inside, and honours `--branch then|else`.
- Budget: `max_segments` (default 20000) raises `_Stop`, which is caught and
  turned into a warning -- the partial timeline is still rendered.

## Invariants you can rely on (and must not break)

- **Segments never overlap per (element, channel).** Cursors are monotonic, so
  `Timeline._build_index` can bisect on `t1`. If you ever add a statement that
  plays two things at once on one element, fix the index first.
- **`Waveform.minmax` is exact**, not point-sampled: a 40 ns Gaussian inside one
  pixel column must not vanish. Keep it that way when adding waveform kinds.
- **`Timeline.peak` adds the DC level inside the abs** (`|level + wf|`), because
  a `ramp_to_zero` on top of a +60 mV hold would otherwise claim 120 mV and
  squash the lane.
- **All renderers go through `axis.TimeAxis`** (`x_of`/`t_of`). Never assume the
  axis is linear: idle stretches are collapsed by default, so sample columns by
  inverse-mapping `t_of(x)`, not by dividing the time span.
- **`Loc` must point at the user's script.** `qua._loc()` walks out of frames
  whose filename is inside the `qua_scope` package directory (not its parent --
  the examples live one level up and were being skipped).

## Traps already hit -- do not re-learn these

- `from qm.qua import *` imports **`__all__` only**, so a module-level
  `__getattr__` fallback does not save you: an unmodelled QUA name raises
  NameError mid-program and truncates the trace. Unmodelled API names must be
  real globals appended to `__all__` (see `_UNMODELLED_API` at the bottom of
  `qua.py`). Add to that list when a script hits a name we do not have.
- Stubs must be **falsy and empty-iterable** (`loader._Any`): scripts commonly
  spin on `while job.result_handles.is_processing():`, which must terminate.
- `elif_`/`else_` attach to the last `If` in the *current block list*
  (`_Builder.last_if` keyed by `id(body)`); `case_`/`default_` attach via
  `_switch_stack`. Do not "simplify" these into the block stack.
- `qualang_tools.loops.from_array` is rewritten to a `for_each_` over the
  concrete array (`qua.ArrayLoop`), which is exact; do not reconstruct
  start/stop/step.
- Declared variable names are recovered from the source line
  (`a = declare(fixed)` -> `a`) and de-duplicated with a `#n` suffix. Labels
  like `x180*amp(a=0.25)` depend on this.
- Axis compression must clamp: hundreds of collapsed gaps (a fully unrolled
  averaging loop) would claim more width than the plot has, so `gap_px` shrinks
  when `gap_px * n > 0.4 * plot_w`.
- SVG is written by hand. Escape text, keep coordinates finite and inside the
  canvas, and re-check with an XML parse after changing markup.

## Tests -- the contract

```bash
cd /home/roland/projects/portal/qua-scope && python3 tests/test_timing.py
```

35 checks over cursors, bare `align`, `amp()`/`duration=`/`truncate=`,
measurement windows, unrolling and skip counts, run-time branches, sticky
levels, frame/frequency updates, expression evaluation, and an end-to-end pass
over `examples/` (every example must load, schedule, render valid SVG, and
render text). Style matches the repo's other suite (`coulomb-detect-py`):
plain script, `[ok ]`/`[FAIL]` lines, non-zero exit on failure. Add a check
whenever you change a semantic -- that is what makes the diagrams trustworthy.

`examples/` has power Rabi (numpy config + `from_array`), Ramsey (frame
rotation, `for_each_`), active reset (run-time branch, `while_`), and a
sticky-gate charge-stability sweep (config inline in the script). Their rendered
`.scope.svg` files are checked in next to them.

## Extending it

- **A statement we do not model**: add an `ir` node, emit it in `qua.py`, add a
  `do_*` in `Interpreter` and an entry in `_DISPATCH`, then a test.
- **Digital markers** are parsed (`Pulse.digital_marker`,
  `Machine.digital_waveforms`) but not drawn -- a thin digital lane under each
  element is the obvious next feature.
- **Grouping lanes by controller port** (what a real scope probes, and it would
  show two elements sharing an output) is not implemented; `Element.ports` has
  what you need.
- **An interactive HTML/artifact renderer** (pan/zoom, hover) was offered to the
  user and not built. `cli.py --json` already emits the whole timeline, so a
  viewer can be a separate consumer rather than a fifth renderer.
- Fixed-point (4.28) rounding is not simulated except where a QUA op truncates
  (`Cast.mul_int_by_fixed`).

## State

Nothing is committed -- the whole `qua-scope/` directory is untracked in the
`portal` repo (which otherwise holds `coulomb-detect-py`). The user has not
asked for a commit; ask before making one.
