# Grammar-first Coulomb-diamond detection (Python)

Detection track (Layers 1-5) for 2D bias spectroscopy in current mode.
numpy-only. Analysis and simulation are SEPARATE modules.

## Files
- `detect_current_mode.py` -- the analysis. Entry point:
  `run_detection(img, vg, vsd, cfg=DetectConfig()) -> DetectResult`.
  Takes any uniform-grid current map; axes in either direction; vsd must
  span through ~0. All thresholds live in `DetectConfig` (data units) --
  see its docstring for which to scale with your addition voltage.
- `synth_current_mode.py` -- the simulation model (dead band, bias-linear
  smearing, AR(1) row noise, clutter, excited-state steps, disorder scatter
  + even/odd alternating spacings). Bit-exact RNG port of the JS demo:
  a given (params, seed) reproduces the same instance.
- `test_golden.py` -- the contract: three suites (baseline / disorder
  scatter / even-odd alternation) x three seeds, asserted against truth.
  Run `python3 test_golden.py`.
- `run_on_data.py` -- template for real scans (.npz with img/vg/vsd keys,
  or adapt the loader). `--demo` runs a synthetic instance.

## Deviations from the JS reference (deliberate)
1. Fits use pixel CENTERS for the vs coordinate; the JS uses pixel top
   edges (half-pixel convention slip preserved there for its own goldens).
   Centers are correct for real data where the vsd axis is sample positions.
2. Vectorized float summation order differs from the JS sequential sums;
   marginal instances at the stress boundary (spacing ratio 1.8) can land
   a few % differently. Golden tolerances are set per regime accordingly.

## Interpreting the readout
- `alpha` = Mp*|Mm|/(Mp+|Mm|), the gate lever arm from the pooled slopes.
- `Ec` per diamond = alpha x apex gap (grammar estimate); `EcRegion` is the
  independent width-collapse cross-check from blockade intervals.
- `tauP/tauM`: hierarchical slope dispersion -- elevated tau is the
  parasitic-dot / mixed-structure diagnostic.
- `minus_via_ladder`: the '-' slope came from the ladder-constrained scan
  against the '+' apex skeleton (weak-channel rescue), not its own
  orientation histogram.
- A correct even/odd Ec RATIO with wrong parity usually means a lost
  edge-of-frame diamond, not wrong physics.

## Known limits (deliberate, shared with the JS reference)
- Strong family hardcoded '+' in the ladder rescue; on real data assign by
  measured channel weight.
- 2x-gap ambiguity (big diamond vs missed apex) is flagged, not resolved;
  the arbiter should be the region track's per-diamond collapse.
- Comb fusion weights and dedupe thresholds are hand-set pending the GLRT.
- Detection only: outputs are geometric hypotheses for the estimation
  track, which fits the physics forward model against raw conditioned data.
