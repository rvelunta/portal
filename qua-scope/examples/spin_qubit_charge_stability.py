"""Sticky gate pulses + RF sensor readout -- a charge-stability sweep.

Self-contained (config in the same file, which the loader also handles). The two
plunger gates are STICKY elements: each step adds to the held level and the
level stays there until the next step or `ramp_to_zero`, which is what makes a
stability-diagram raster look like a staircase rather than a pulse train.
"""

from qm.qua import *

n_avg = 2
n_x = 5                      # fast axis points (gate 2)
n_y = 3                      # slow axis points (gate 1)
step_v = 0.02                # V per step
readout_len = 1000           # ns

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {1: {"offset": 0.0}, 2: {"offset": 0.0},
                               3: {"offset": 0.0}, 4: {"offset": 0.0}},
            "analog_inputs": {1: {"offset": 0.0}},
        }
    },
    "elements": {
        "gate_1": {
            "singleInput": {"port": ("con1", 1)},
            "operations": {"step": "step_pulse"},
            "hold_offset": {"duration": 1},          # sticky
        },
        "gate_2": {
            "singleInput": {"port": ("con1", 2)},
            "operations": {"step": "step_pulse"},
            "hold_offset": {"duration": 1},          # sticky
        },
        "sensor": {
            "mixInputs": {"I": ("con1", 3), "Q": ("con1", 4),
                          "lo_frequency": 0.0, "mixer": "mx"},
            "intermediate_frequency": 150e6,
            "operations": {"readout": "readout_pulse"},
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 300,
            "smearing": 40,
        },
    },
    "pulses": {
        "step_pulse": {"operation": "control", "length": 16,
                       "waveforms": {"single": "step_wf"}},
        "readout_pulse": {"operation": "measurement", "length": readout_len,
                          "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
                          "integration_weights": {"cos": "w_cos",
                                                  "sin": "w_sin"}},
    },
    "waveforms": {
        "step_wf": {"type": "constant", "sample": step_v},
        "readout_wf": {"type": "constant", "sample": 0.05},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "integration_weights": {
        "w_cos": {"cosine": [(1.0, readout_len)], "sine": [(0.0, readout_len)]},
        "w_sin": {"cosine": [(0.0, readout_len)], "sine": [(1.0, readout_len)]},
    },
    "mixers": {"mx": [{"intermediate_frequency": 150e6, "lo_frequency": 0.0,
                       "correction": [1.0, 0.0, 0.0, 1.0]}]},
}

with program() as charge_stability:
    n = declare(int)
    i = declare(int)
    j = declare(int)
    I = declare(fixed)
    Q = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(i, 0, i < n_y, i + 1):
            play("step", "gate_1")                 # slow axis: one step per row
            with for_(j, 0, j < n_x, j + 1):
                play("step", "gate_2")             # fast axis: staircase
                align()
                measure("readout", "sensor", None,
                        demod.full("cos", I, "out1"),
                        demod.full("sin", Q, "out1"))
                wait(100, "gate_1", "gate_2")
            ramp_to_zero("gate_2")                 # back to the start of the row
        ramp_to_zero("gate_1")
