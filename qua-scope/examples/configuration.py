"""A conventional two-element QM config: one driven qubit, one readout
resonator, both on mixed inputs. Written the way QM example configs are, so it
exercises the loader (numpy waveform construction, unit helpers, mixers)."""

import numpy as np

# ------------------------------------------------------------------ parameters
qubit_IF = 50e6
qubit_LO = 4.8e9
resonator_IF = 60e6
resonator_LO = 7.1e9

pi_len = 40                # ns
pi_amp = 0.25              # V
pi_half_amp = pi_amp / 2
sigma = pi_len / 5

readout_len = 800          # ns
readout_amp = 0.08
time_of_flight = 200       # ns
smearing = 0

saturation_len = 2000
saturation_amp = 0.1


def gaussian(amplitude, length, sigma):
    t = np.linspace(0, length - 1, length)
    wf = amplitude * np.exp(-((t - (length - 1) / 2) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in wf]


def gaussian_derivative(amplitude, length, sigma):
    t = np.linspace(0, length - 1, length)
    c = (length - 1) / 2
    wf = amplitude * (-(t - c) / sigma ** 2) * np.exp(
        -((t - c) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in wf]


pi_wf = gaussian(pi_amp, pi_len, sigma)
pi_der_wf = gaussian_derivative(0.02, pi_len, sigma)     # DRAG quadrature
pi_half_wf = gaussian(pi_half_amp, pi_len, sigma)


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    n = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(n * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]


# ------------------------------------------------------------------ config
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},   # qubit I
                2: {"offset": 0.0},   # qubit Q
                3: {"offset": 0.0},   # resonator I
                4: {"offset": 0.0},   # resonator Q
            },
            "digital_outputs": {1: {}},
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},
                2: {"offset": 0.0, "gain_db": 0},
            },
        }
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "x180": "pi_pulse",
                "x90": "pi_half_pulse",
                "saturation": "saturation_pulse",
            },
        },
        "resonator": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {"readout": "readout_pulse"},
            "outputs": {"out1": ("con1", 1), "out2": ("con1", 2)},
            "time_of_flight": time_of_flight,
            "smearing": smearing,
        },
    },
    "pulses": {
        "pi_pulse": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "pi_wf", "Q": "pi_der_wf"},
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "pi_half_wf", "Q": "zero_wf"},
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms": {"I": "saturation_wf", "Q": "zero_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
            "integration_weights": {"cos": "cosine_weights",
                                    "sin": "sine_weights",
                                    "minus_sin": "minus_sine_weights"},
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "saturation_wf": {"type": "constant", "sample": saturation_amp},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "pi_wf": {"type": "arbitrary", "samples": pi_wf},
        "pi_der_wf": {"type": "arbitrary", "samples": pi_der_wf},
        "pi_half_wf": {"type": "arbitrary", "samples": pi_half_wf},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "cosine_weights": {"cosine": [(1.0, readout_len)],
                           "sine": [(0.0, readout_len)]},
        "sine_weights": {"cosine": [(0.0, readout_len)],
                         "sine": [(1.0, readout_len)]},
        "minus_sine_weights": {"cosine": [(0.0, readout_len)],
                               "sine": [(-1.0, readout_len)]},
    },
    "mixers": {
        "mixer_qubit": [{"intermediate_frequency": qubit_IF,
                         "lo_frequency": qubit_LO,
                         "correction": IQ_imbalance(0.0, 0.0)}],
        "mixer_resonator": [{"intermediate_frequency": resonator_IF,
                             "lo_frequency": resonator_LO,
                             "correction": IQ_imbalance(0.0, 0.0)}],
    },
}
