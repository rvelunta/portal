"""Power Rabi: sweep the drive amplitude, read out, average.

Ordinary QUA -- nothing here knows about qua-scope."""

import numpy as np
from qm import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array

from configuration import config

n_avg = 100
amps = np.arange(0.0, 1.0, 0.25)          # amplitude scale factors
thermal_decay = 10_000                     # ns

with program() as power_rabi:
    n = declare(int)
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amps)):
            play("x180" * amp(a), "qubit")
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
            wait(thermal_decay // 4, "qubit")

    with stream_processing():
        I_st.buffer(len(amps)).average().save("I")
        Q_st.buffer(len(amps)).average().save("Q")

qmm = QuantumMachinesManager(host="192.168.1.1", port=80)
qm = qmm.open_qm(config)
job = qm.execute(power_rabi)
