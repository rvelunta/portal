"""Ramsey with a virtual-Z detuning: x90 - tau - (frame rotation) - x90 - read."""

from qm.qua import *
from qm import QuantumMachinesManager

from configuration import config

n_avg = 20
taus = [16, 40, 100, 200, 400]      # ns
detuning = 1e6                       # Hz, applied as a frame rotation

with program() as ramsey:
    n = declare(int)
    tau = declare(int)
    phase = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        with for_each_(tau, [t // 4 for t in taus]):
            reset_frame("qubit")
            play("x90", "qubit")
            wait(tau, "qubit")
            assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, tau * 4))
            frame_rotation_2pi(phase, "qubit")
            play("x90", "qubit")
            align("qubit", "resonator")
            measure("readout", "resonator", None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q))
            wait(2500, "qubit", "resonator")

qmm = QuantumMachinesManager(host="192.168.1.1")
qm = qmm.open_qm(config)
job = qm.execute(ramsey)
