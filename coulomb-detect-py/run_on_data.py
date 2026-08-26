"""Run the detector on your own measured data.

Expected input: a 2D current map with its two voltage axes. The vsd axis must
span through ~0 (the dead band / apex line inside the scan). Axes may run in
either direction. Works from .npz, .npy + axes, or adapt the loader to your
HDF5 layout.

Usage:
    python3 run_on_data.py scan.npz            # expects keys: img, vg, vsd
    python3 run_on_data.py --demo              # synthetic instance instead
"""
import sys
import numpy as np
from detect_current_mode import run_detection, DetectConfig


def load(path):
    d = np.load(path)
    return d["img"], d["vg"], d["vsd"]


def main():
    if "--demo" in sys.argv:
        from synth_current_mode import SynthParams, synth_current
        img, vg, vsd, _ = synth_current(SynthParams(spacing_ratio=1.35,
                                                    spacing_scatter=0.10), 7)
    else:
        img, vg, vsd = load(sys.argv[1])

    # thresholds in DetectConfig are in your voltage units, tuned for apex
    # spacings ~0.1-0.3 V; scale the members marked [scale with dVg] in the
    # class docstring if your addition voltages differ substantially, e.g.:
    #   cfg = DetectConfig(apex_nms_vg=0.007, apex_cluster_vg=0.003,
    #                      min_support_vg=0.0035, ...)   # ~10x smaller dVg
    cfg = DetectConfig()

    r = run_detection(img, vg, vsd, cfg)
    print(r.summary())
    print(f"  tau+ {r.tauP:.4f}  tau- {r.tauM:.4f}   (slope-dispersion diagnostics)")
    for g in r.EcPerDiamond:
        flag = "" if g["resolved"] else "  [unresolved gap]"
        ec = f"{1e3*g['Ec']:.1f} mV" if np.isfinite(g["Ec"]) else "--"
        print(f"  diamond [{g['vgL']:.3f}, {g['vgR']:.3f}] V : Ec = {ec}{flag}")


if __name__ == "__main__":
    main()
