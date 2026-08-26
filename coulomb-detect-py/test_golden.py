"""Golden tests: synthesis -> detection, asserted against ground truth.

The contract is truth-tolerance, not bit-parity with the JS reference: the
Python analysis fits against pixel CENTERS (correct for real data), while the
JS fits against pixel edges (a half-pixel convention bug preserved there for
its own goldens). Tolerances below match the demonstrated JS performance.

Run: python3 test_golden.py
"""

import math
import numpy as np
from synth_current_mode import SynthParams, synth_current, true_ec_per_diamond
from detect_current_mode import run_detection

SEEDS = [7, 42, 1234]
ALPHA_TRUE = 0.35 * 0.55 / (0.35 + 0.55)

GENTLE = dict(noise=0.04, rowNoise=0.05, clutter=2, tau=0.04, Vdead=0.007, smear=0.12)
STATED = dict(noise=0.07, rowNoise=0.10, clutter=4, tau=0.06, Vdead=0.009, smear=0.22)

failures = []


def check(name, cond, msg=""):
    status = "ok " if cond else "FAIL"
    print(f"    [{status}] {name} {msg}")
    if not cond:
        failures.append(f"{name} {msg}")


def match_diamonds(est, tru, tol=0.06):
    errs = []
    for g in est:
        if not (g["resolved"] and np.isfinite(g["Ec"])):
            continue
        cg = (g["vgL"] + g["vgR"]) / 2
        t = min(tru, key=lambda t: abs((t["vgL"] + t["vgR"]) / 2 - cg))
        if abs((t["vgL"] + t["vgR"]) / 2 - cg) < tol:
            errs.append(abs(g["Ec"] / t["Ec"] - 1))
    return errs


def run(tag, params, seed):
    img, vg, vsd, truth = synth_current(params, seed)
    r = run_detection(img, vg, vsd)
    print(f"  [{tag} s={seed}] {r.summary()}")
    return r, truth


def suite_baseline():
    print("== baseline (even spacing) ==")
    for tag, cfg, a_tol in (("gentle", GENTLE, 0.05), ("stated", STATED, 0.06)):
        for seed in SEEDS:
            r, truth = run(tag, SynthParams(**cfg), seed)
            check("alpha", abs(r.alpha / ALPHA_TRUE - 1) <= a_tol,
                  f"err {(r.alpha/ALPHA_TRUE-1)*100:+.1f}%")
            check("delta", abs(r.delta - truth["delta"]) <= 0.005,
                  f"{1e3*abs(r.delta-truth['delta']):.1f} mV")
            check("Vdead", abs(r.Vdead - cfg["Vdead"]) <= 0.004,
                  f"{1e3*abs(r.Vdead-cfg['Vdead']):.1f} mV")
            errs = match_diamonds(r.EcPerDiamond, true_ec_per_diamond(truth))
            check("Ec spectrum", len(errs) >= 2 and np.median(errs) <= 0.20,
                  f"{len(errs)} matched, med err {100*np.median(errs) if errs else 0:.0f}%")


def suite_scatter():
    print("== disorder scatter ==")
    for scat in (0.10, 0.18):
        for seed in SEEDS:
            p = SynthParams(spacing_scatter=scat, **STATED)
            r, truth = run(f"scat={scat}", p, seed)
            check("alpha", abs(r.alpha / ALPHA_TRUE - 1) <= 0.08,
                  f"err {(r.alpha/ALPHA_TRUE-1)*100:+.1f}%")
            errs = match_diamonds(r.EcPerDiamond, true_ec_per_diamond(truth))
            check("Ec spectrum", len(errs) >= 2 and np.median(errs) <= 0.25,
                  f"{len(errs)} matched, med err {100*np.median(errs) if errs else 0:.0f}%")


def suite_alternation():
    print("== even/odd alternation ==")
    # r=1.8 is the stress boundary: narrow diamonds crowd under smear and the
    # JS reference also degrades there (-10..-14%); tolerance reflects that
    for ratio, a_tol in ((1.4, 0.12), (1.8, 0.20)):
        for seed in SEEDS:
            p = SynthParams(spacing_ratio=ratio, **STATED)
            r, truth = run(f"ratio={ratio}", p, seed)
            check("alpha", abs(r.alpha / ALPHA_TRUE - 1) <= a_tol,
                  f"err {(r.alpha/ALPHA_TRUE-1)*100:+.1f}%")
            errs = match_diamonds(r.EcPerDiamond, true_ec_per_diamond(truth))
            check("Ec spectrum", len(errs) >= 2 and np.median(errs) <= 0.25,
                  f"{len(errs)} matched, med err {100*np.median(errs) if errs else 0:.0f}%")


if __name__ == "__main__":
    suite_baseline()
    suite_scatter()
    suite_alternation()
    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("all golden tests passed")
