import numpy as np

from time import sleep
from src.hmp40x0 import DriverHMP40X0

if __name__ == '__main__':
    COM_PORT = "COM6"
    HMP4030 = DriverHMP40X0(COM_PORT)
    HMP4030.start_serial()

    # Arbitrary waveform
    dly = 100e-3
    time = np.linspace(0, 10, 128)
    voltage = 8 + 8 * np.sin(2 * np.pi * time/128)
    current = np.zeros(shape=voltage.shape) + 100e-3

    # Protocol to test
    HMP4030.do_reset()
    HMP4030.do_check_idn()

    HMP4030.ch_set_parameter(0, 0, 1e-3)
    HMP4030.ch_set_parameter(1, 10.5, 100e-3)
    HMP4030.ch_set_parameter(2, 0, 10e-3)
    HMP4030.afg_set_waveform(2, voltage, current, dly + time, 1)

    HMP4030.output_activate()
    sleep(1)
    HMP4030.afg_start()

    for ite in range(0, 100):
        HMP4030.ch_read_parameter(2)
        sleep(1)

    HMP4030.afg_stop()
    HMP4030.output_deactivate()