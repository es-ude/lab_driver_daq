import numpy as np
from time import sleep
import pyvisa


def scan_instruments(do_print=True) -> list:
    """Scanning the VISA bus for instruments"""
    rm = pyvisa.ResourceManager()
    obj_inst = rm.list_resources()

    out_dev_adr = list()
    for idx, inst_name in enumerate(obj_inst):
        if idx == 0 and do_print:
            print(f"\nUsing VISA driver: {rm}")
            print("Available devices")
            print("--------------------------------------")
        elif do_print:
            print(f"{idx}: {inst_name}")
        out_dev_adr.append(inst_name)
    return out_dev_adr


class DriverMXO4X:
    """Class for handling the Rohde and Schwarz Mixed-Signal Oscilloscope MXO44 in Python"""
    SerialDevice: pyvisa.Resource
    SerialActive = False
    _device_name_chck = "MXO"

    def __init__(self):
        pass

    def __write_to_dev(self, order: str) -> None:
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        text_out = self.SerialDevice.query(order)
        return text_out

    def __init_dev(self, do_reset=True):
        """"""
        if self.SerialActive:
            if do_reset:
                self.do_reset()
            self.__write_to_dev("SYST:MIX")
            print(f"Right device is selected with: {self.get_id(False)}")
        else:
            print("Not right selected device. Please check!")

    def __do_check_idn(self) -> None:
        """Checking the IDN"""
        id_back = self.get_id(False)
        if self._device_name_chck in id_back:
            self.SerialActive = True
        else:
            self.SerialActive = False

    def start_serial_known_target(self, resource_name: str, do_reset=False) -> None:
        """Open the serial connection to device"""
        rm = pyvisa.ResourceManager()
        self.SerialDevice = rm.open_resource(resource_name)

        self.__do_check_idn()
        self.__init_dev(do_reset)

    def start_serial(self, do_reset=False) -> None:
        """Open the serial connection to device"""
        list_dev = scan_instruments()

        for inst_name in list_dev:
            rm = pyvisa.ResourceManager()
            self.SerialDevice = rm.open_resource(inst_name)
            self.__do_check_idn()
            if self.SerialActive:
                break

        self.__init_dev(do_reset)

    def close_serial(self) -> None:
        """Closing the serial connection"""
        self.SerialDevice.close()
        self.SerialActive = False

    def get_id(self, do_print=True) -> str:
        """Getting the device ID"""
        id = self.__read_from_dev("*IDN?")
        if do_print:
            print(id)
        return id

    def do_beep(self, num_iterations=1) -> None:
        """Doing a single beep on device"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            for ite in range(0, num_iterations):
                self.__write_to_dev("SYST:BEEP")
                sleep(1)

    def do_reset(self) -> None:
        """Reset the device"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev("*RST")
            sleep(2)
            self.do_beep()


if __name__ == "__main__":
    scan_instruments()

    inst0 = DriverMXO4X()
    inst0.start_serial()
    inst0.do_beep()
