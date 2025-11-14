import numpy as np
import pyvisa
from time import sleep
from logging import getLogger, Logger
from serial.tools import list_ports
from lab_driver.scan_instruments import scan_instruments


class DriverHMP40X0:
    SerialDevice: pyvisa.Resource
    _device_name_chck = "HMP"
    _logger: Logger
    _usb_vid = 0x0403
    _usb_pid = 0xed72

    def __init__(self, num_ch: int=4) -> None:
        """Class for Remote Controlling the Power Supply R&S HMP40X0 via USB
        :param num_ch:  Number of available device channels (HMP4030 = 3, HMP4040 = 4)
        :return:        None
        """
        self._logger = getLogger(__name__)
        self.__NoChannels = num_ch
        self.SerialActive = False
        self.ChannelUsed = [False, False, False, False]
        self.SetCH = [ChannelInfoHMP40X0(idx) for idx in range(num_ch)]

    def __write_to_dev(self, order: str) -> None:
        """Writing content to serial connection
        Args:
            order:  String for content
        Return:
            None
        """
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        """Reading content from serial connection
        Args:
            order:  String for content
        Return:
            String with information from device
        """
        text_out = self.SerialDevice.query(order)
        return text_out

    def __init_dev(self, do_reset=True) -> None:
        """Initialization of device
        Args:
            do_reset:       Doing a reset of device
        Return:
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
        if self._device_name_chck in id_back:
            self.SerialActive = True
        else:
            self.SerialActive = False

    def serial_open_known_target(self, resource_name: str, do_reset=False) -> None:
        """Open the serial connection to device
        Args:
            resource_name:  Ressource name of serial communication from device
            do_reset:       Doing a device reset
        Return:
            None
        """
        rm = pyvisa.ResourceManager()
        self.SerialDevice = rm.open_resource(resource_name)

        self.__do_check_idn()
        self.__init_dev(do_reset)

    def get_usb_vid(self):
        return self._usb_vid

    def get_usb_pid(self):
        return self._usb_pid

    def scan_com_name(self) -> list:
        """Returning the COM Port name of the addressable devices"""
        available_coms = list_ports.comports()
        list_right_com = [port.device for port in available_coms if
                          port.vid == self._usb_vid and port.pid == self._usb_pid]
        if len(list_right_com) == 0:
            errmsg = '\n'.join([f"{port.usb_description()} {port.device} {port.usb_info()}" for port in available_coms])
            raise ConnectionError(f"No COM Port with right USB found - Please adapt the VID and PID values from "
                                  f"available COM ports:\n{errmsg}")
        self._logger.debug(f"Found {len(list_right_com)} COM ports available")
        return list_right_com

    def serial_open(self, do_reset=False) -> None:
        """Open the serial connection to device"""
        list_dev = scan_instruments(self)
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

    def ch_read_parameter(self, sel_ch: int) -> [float, float]:
        """Read sense parameter from HMP40X0"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            text = list()
            text.append(self.SetCH[sel_ch].sel_ch())
            text.append(self.SetCH[sel_ch].read_ch_voltage())
            text.append(self.SetCH[sel_ch].read_ch_current())
            # Getting the information
            self.__write_to_dev(text[0])
            volt = float(self.__read_from_dev(text[1]))
            curr = float(self.__read_from_dev(text[2]))
            # Right-back information
            self.SetCH[sel_ch].set_sense_parameter(volt, curr)

            return volt, curr

    def output_activate(self) -> None:
        """Activate the already setted channels of HMP40X0"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev("OUTP:GEN ON")
            self.do_beep()

    def output_deactivate(self) -> None:
        """Deactivate all channels of HMP40X0"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev("OUTP:GEN OFF")

            for idx in range(0, self.__NoChannels):
                text = "INST OUT " + str(idx + 1)
                self.__write_to_dev(text)
                self.__write_to_dev("OUTP OFF")
            self.do_beep()

    def afg_start(self) -> None:
        """Starting the Arbitrary Generator on the configured channel"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            text = str()
            for idx in range(self.__NoChannels):
                if self.SetCH[idx].ChannelUsedAFG:
                    text = self.SetCH[idx].set_arbitrary_start()
            self.__write_to_dev(text)

    def afg_stop(self) -> None:
        """Stopping the Arbitrary Generator on the configured channel"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            text = str()
            for idx in range(self.__NoChannels):
                if self.SetCH[idx].ChannelUsedAFG:
                    text = self.SetCH[idx].set_arbitrary_stop()
            self.__write_to_dev(text)

    def ch_set_parameter(self, sel_ch: int, volt: float, cur: float) -> None:
        """Set parameters I/V of each channel"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            text = list()
            text.append(self.SetCH[sel_ch].sel_ch())
            text.append(self.SetCH[sel_ch].set_ch_voltage(volt))
            text.append(self.SetCH[sel_ch].set_ch_current_limit(cur))
            text.append(self.SetCH[sel_ch].set_ch_output())

            # Configuring the channels
            for string in text:
                self.__write_to_dev(string)

    def afg_set_waveform(self, sel_ch: int, voltage: np.ndarray, current: np.ndarray, time: np.ndarray, num_cycles=0):
        """Set arbitrary waveform of the selected channel (max. 128 points)"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            text = list()
            text.append(self.SetCH[sel_ch].set_arbitrary_waveform(voltage, current, time))
            text.append(self.SetCH[sel_ch].set_arbitrary_repetition(num_cycles))
            text.append(self.SetCH[sel_ch].set_arbitrary_transfer())

            # Configuring the channels
            for string in text:
                self.__write_to_dev(string)


class ChannelInfoHMP40X0:
    def __init__(self, ch_no: int):
        """"""
        self.ChannelActive = False
        self.ChannelUsedAFG = False
        self.ChannelNumber = 1 + ch_no

        self.VoltageSet = 0
        self.VoltageSense = 0
        self.CurrentLimit = 0
        self.CurrentSense = 0
        self.PowerSense = 0

        self.__max_voltage = 32.05
        self.__min_current = 0
        self.__max_current = 10
        self.__min_voltage = 0
        self.__dec_voltage = 3
        self.__dec_current = 3
        self.__dec_time = 2

    def sel_ch(self) -> str:
        """"""
        return "INST OUT" + str(self.ChannelNumber)

    def set_ch_voltage(self, voltage_value: float) -> str:
        """"""
        self.VoltageSet = voltage_value
        return "VOLT " + str(voltage_value)

    def set_ch_current_limit(self, current_value: float) -> str:
        """"""
        self.CurrentLimit = current_value
        return "CURR " + str(current_value)

    def get_ch_parameter(self) -> str:
        """"""
        return "APPL?"

    def set_ch_output(self) -> str:
        """"""
        return "OUTP:SEL 1"

    def read_ch_voltage(self) -> str:
        """"""
        return "MEAS:VOLT?"

    def read_ch_current(self) -> str:
        """"""
        return "MEAS:CURR?"

    def set_sense_parameter(self, voltage: float, current: float) -> None:
        """"""
        self.VoltageSense = voltage
        self.CurrentSense = current
        self.PowerSense = voltage * current
        print(f"Sense: {voltage: .3f} V - {current: .5f} A - {self.PowerSense: .6f} W")

    def set_arbitrary_waveform(self, voltage: np.ndarray, current: np.ndarray, time: np.ndarray) -> str:
        """Defining the Arbitrary Waveform for Powr Supply (max. 128 points)"""
        text = "ARB:DATA "
        size_list = min(voltage.size, current.size, time.size, 128)

        # Pre-Processing
        voltage_in = np.round(voltage, self.__dec_voltage)
        current_in = np.round(current, self.__dec_current)
        time_in = np.round(time, self.__dec_time)

        # Value clipping
        if np.max(voltage_in) > self.__max_voltage:
            posx = np.where(voltage_in > self.__max_voltage)
            voltage_in[posx] = self.__max_voltage
        if np.min(voltage_in) < self.__min_voltage:
            posx = np.where(voltage_in < self.__min_voltage)
            voltage_in[posx] = self.__min_voltage

        if np.max(current_in) > self.__max_voltage:
            posx = np.where(current_in > self.__max_current)
            current_in[posx] = self.__max_voltage
        if np.min(current_in) < self.__min_voltage:
            posx = np.where(current_in < self.__min_current)
            current_in[posx] = self.__min_voltage

        # Text generation
        for ite in range(0, size_list):
            text = text + str(voltage_in[ite]) + "," + str(current_in[ite]) + "," + str(time_in[ite])
            if ite < size_list - 1:
                text = text + ","

        self.ChannelActive = True
        self.ChannelUsedAFG = True
        return text

    def set_arbitrary_repetition(self, num_cycles: int) -> str:
        """Setting the cycles numbers of arbitrary waveform (0= infinitely)"""
        cycles_num = min(num_cycles, 255)
        return "ARB:REP " + str(cycles_num)

    def set_arbitrary_transfer(self) -> str:
        """"""
        return "ARB:TRAN " + str(self.ChannelNumber)

    def set_arbitrary_start(self) -> str:
        """"""
        return "ARB:START " + str(self.ChannelNumber)

    def set_arbitrary_stop(self) -> str:
        return "ARB:STOP " + str(self.ChannelNumber)
