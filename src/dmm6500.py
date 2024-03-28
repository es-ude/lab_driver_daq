import numpy as np
import pyvisa


def scan_instruments(get_id=False) -> None:
    """Scanning the VISA bus for instruments"""
    rm = pyvisa.ResourceManager()
    print(f"\nAvailable VISA driver: {rm}")
    print("\nAvailable devices")
    print("--------------------------------------")
    obj_inst = rm.list_resources()
    for idx, inst_name in enumerate(obj_inst):
        if get_id:
            inst0 = rm.open_resource(inst_name)
            id = " --> " + inst0.query("*IDN?")
            for ite in range(4):
                inst0.write("SYST:BEEP")
                sleep(0.5)
            inst0.close()
        else:
            id = ""
        print(f"{idx}: {inst_name}{id}")


class dmm6500:
    """Class for handling the Keithley Digital Multimeter 6500 in Python"""

    def __init__(self):
        pass
    