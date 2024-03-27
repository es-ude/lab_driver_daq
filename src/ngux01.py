import numpy as np
from time import sleep
import pyvisa


def scan_instruments() -> None:
    """Scanning the VISA bus for instruments"""
    rm = pyvisa.ResourceManager()
    print("\nAvailable VISA driver:")
    print(rm)
    print("\nAvailable devices")
    print("--------------------------------------")
    obj_inst = rm.list_resources()
    for idx, inst_name in enumerate(obj_inst):
        inst0 = rm.open_resource(inst_name)
        id = inst0.query("*IDN?")
        for ite in range(4):
            inst0.write("SYST:BEEP")
            sleep(0.5)
        inst0.close()
        print(f"{idx}: {inst_name} --> {id}")


class NGUX01:
    """Class for handling the Rohde and Schwarz Sourcemeter NGUX01 in Python"""
    SerialDevice: pyvisa.Resource
    SerialActive = False

    def __init__(self):
        pass

    def __write_to_dev(self, order: str) -> None:
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        text_out = self.SerialDevice.query(order)
        return text_out

    def start_serial(self, resource_name: str):
        """Open the serial connection to HMP40X0"""
        rm = pyvisa.ResourceManager()
        self.SerialDevice = rm.open_resource(resource_name)
        self.SerialActive = True

        self.do_reset()
        self.__write_to_dev("SYST:MIX")
        self.do_check_idn()

    def close_serial(self):
        """Closing the serial connection"""
        self.SerialDevice.close()
        self.SerialActive = False

    def do_reset(self) -> None:
        """Reset the device"""
        self.__write_to_dev("*RST")
        sleep(5)
        self.do_beep()

    def do_check_idn(self) -> None:
        """Checking the IDN"""
        self.ID_Device = self.__read_from_dev("*IDN?")
        print(self.ID_Device)

    def do_beep(self) -> None:
        """Doing a single beep on device"""
        for ite in range(0, 1):
            self.__write_to_dev("SYST:BEEP")
            sleep(2)


if __name__ == "__main__":
    scan_instruments()

    sleep(1)
    inst0 = NGUX01()
    inst0.start_serial('ASRL6::INSTR')
    inst0.do_beep()
