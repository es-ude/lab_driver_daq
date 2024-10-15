import numpy as np
from time import sleep
import pyvisa
import platform


def scan_instruments(do_print=True) -> list:
    """Scanning the VISA bus for instruments
    Args:
        do_print: True to print every detected instrument
    Returns:
        List of all detected instruments
    """
    rm = pyvisa.ResourceManager()
    obj_inst = list(rm.list_resources())
    if platform.system() == "Linux":    # TODO: can't find device on Linux
        obj_inst = filter(lambda inst_name: "ttyS" not in inst_name, obj_inst)

    out_dev_adr = list()
    for idx, inst_name in enumerate(obj_inst):
        out_dev_adr.append(inst_name)
        # --- Printing the stuff
        if do_print:
            if idx == 0:
                print(f"\nUsing VISA driver: {rm}")
                print("Available devices")
                print("--------------------------------------")
            print(f"{idx}: {inst_name}")
    return out_dev_adr


class DriverMXO4X:
    """Class for handling the Rohde and Schwarz Mixed-Signal Oscilloscope MXO44 in Python"""
    SerialDevice: pyvisa.Resource
    SerialActive = False
    _device_name_chck = "MXO"

    def __init__(self):
        pass

    def __write_to_dev(self, order: str) -> None:
        """Wrapper for executing commands on device
        Args:
            order: command to run on device (may alter device state)
        Returns:
            None
        """
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        """Wrapper for querying data from device
        Args:
            order: command to run on device
        Returns:
            Queried data as a string
        """
        text_out = self.SerialDevice.query(order)
        return text_out

    def __init_dev(self, do_reset=True):
        """If the correct device is selected, initialise it and optionally do a reset
        Args:
            do_reset: reset device or not
        Returns:
            None
        """
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
        self.SerialActive = self._device_name_chck in id_back

    def serial_open_known_target(self, resource_name: str, do_reset=False) -> None:
        """Open the serial connection to device
        Args:
            resource_name: name of the device
            do_reset: reset device during initialisation
        Returns:
            None
        """
        rm = pyvisa.ResourceManager()
        self.SerialDevice = rm.open_resource(resource_name)

        self.__do_check_idn()
        self.__init_dev(do_reset)

    def serial_start(self, do_reset=False) -> None:
        """Open the serial connection to device if it is found
        Args:
            do_reset: reset device during initialisation
        Returns:
            None
        """
        list_dev = scan_instruments(do_print=False)
        rm = pyvisa.ResourceManager()

        # --- Checking if device address is right
        for inst_name in list_dev:
            self.SerialDevice = rm.open_resource(inst_name)
            self.__do_check_idn()
            if self.SerialActive:
                break
            else:
                self.serial_close()

        # --- Init of device
        self.__init_dev(do_reset)

    def serial_close(self) -> None:
        """Close the serial connection
        Args:
            N/A
        Returns:
            None
        """
        self.SerialDevice.close()
        self.SerialActive = False

    def get_id(self, do_print=True) -> str:
        """Getting the device ID
        Args:
            do_print: optionally print the device ID to stdout
        Returns:
            Device ID as a string
        """
        id = self.__read_from_dev("*IDN?")
        if do_print:
            print(id)
        return id

    def do_reset(self) -> None:
        """Reset the device, then wait two seconds
        Args:
            N/A
        Returns:
            None
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev("*RST")
            sleep(2)

    def change_display_mode(self, show_display: bool) -> None:
        """Decide whether display is shown during remote control
        Args:
            show_display: True to show display, False to show static image (may improve performance)
        Returns:
            None
        """
        self.__write_to_dev(f"SYST:DISP:UPD {int(show_display)}")

    def change_remote_text(self, text: str) -> None:
        """Display an additional text in remote control
        Args:
            text: text to display
        Returns:
            None
        """
        self.__write_to_dev(f"SYST:DISP:MESS:STAT ON")
        self.__write_to_dev(f"SYST:DISP:MESS {text}")


if __name__ == "__main__":
    scan_instruments()

    inst0 = DriverMXO4X()
    inst0.serial_start()
    inst0.get_id()

    inst0.do_reset()
    sleep(10)

    inst0.change_display_mode(False)
    inst0.change_remote_text("Hello World!")

    inst0.serial_close()
