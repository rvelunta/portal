"""Active reset then a pi pulse: the branch depends on a measurement result,
which nothing can know before the job runs. qua-scope draws the branch it
would take with the demodulation result unknown and marks it run-time."""

from qm.qua import *
from qm import QuantumMachinesManager

from configuration import config

threshold = 0.002
max_tries = 3

with program() as active_reset:
    n = declare(int)
    tries = declare(int)
    I = declare(fixed)
    Q = declare(fixed)

    with for_(n, 0, n < 10, n + 1):
        # --- reset the qubit until it reads out in |g>
        assign(tries, 0)
        with while_(tries < max_tries):
            measure("readout", "resonator", None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q))
            align("resonator", "qubit")
            with if_(I > threshold):
                play("x180", "qubit")
                assign(tries, tries + 1)
            with else_():
                assign(tries, max_tries)
            align("qubit", "resonator")

        # --- the actual experiment
        play("x180", "qubit")
        align("qubit", "resonator")
        measure("readout", "resonator", None,
                dual_demod.full("cos", "out1", "sin", "out2", I))
        wait(1000, "qubit", "resonator")

qmm = QuantumMachinesManager(host="192.168.1.1")
qm = qmm.open_qm(config)
job = qm.execute(active_reset)
