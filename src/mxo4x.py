import numpy as np
from time import sleep
from src.usb_serial import com_usb

class DriverMXO4X:
    """Class for Remote Controlling the Mixed Signal Osciloscpe R&S MXO4x via USB"""

    def __init__(self, com_name: str, num_ch=3):
        """Initizialization of Power Supply Device with BAUD=9600(com_name, num_of_channels)"""
        self.__NoChannels = num_ch
        self.SerialDevice = com_usb(com_name, 9600)
        self.SerialActive = False
        self.ChannelUsed = [False, False, False, False]
        self.ID_Device = str()

    def __write_to_dev(self, text: str) -> None:
        self.SerialDevice.write_wofb(bytes(text + '\n', 'utf-8'))

    def __read_from_dev(self, text: str) -> str:
        text_out = str(self.SerialDevice.write_wfb_lf(bytes(text + '\n', 'utf-8')), 'UTF-8')
        return text_out

    def start_serial(self):
        """Open the serial connection to MXO4X"""
        self.SerialDevice.setup_usb()

        self.SerialDevice.open()
        self.SerialActive = True

        self.__write_to_dev("SYST:MIX")

    def close_serial(self):
        """Closing the serial connection to HMP4030"""
        self.SerialDevice.close()
        self.SerialActive = False

    def do_reset(self) -> None:
        """Reset the MXO44"""
        self.__write_to_dev("*RST")
        sleep(5)

    def do_check_idn(self) -> None:
        """Checking the IDN of MXO44"""
        # returned "Rohde&Schwarz,<device type>,<part number>/<serial number>,<firmware version>"
        self.ID_Device = self.__read_from_dev("*IDN?")
        print(self.ID_Device)