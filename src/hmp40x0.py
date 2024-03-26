import numpy as np
from time import sleep
from src.usb_serial import com_usb


class DriverHMP40X0:
    """Class for Remote Controlling the Power Supply R&S HMP40X0 via USB"""

    def __init__(self, com_name: str, num_ch=3):
        """Initizialization of Power Supply Device with BAUD=9600(com_name, num_of_channels)"""
        self.__NoChannels = num_ch
        self.SerialDevice = com_usb(com_name, 9600)
        self.SerialActive = False
        self.ChannelUsed = [False, False, False, False]
        self.ID_Device = str()
        self.SetCH1 = ChannelInfoHMP40X0(0)
        self.SetCH2 = ChannelInfoHMP40X0(1)
        self.SetCH3 = ChannelInfoHMP40X0(2)
        self.SetCH4 = ChannelInfoHMP40X0(3)

    def __write_to_dev(self, text: str) -> None:
        self.SerialDevice.write_wofb(bytes(text + '\n', 'utf-8'))

    def __read_from_dev(self, text: str) -> str:
        text_out = str(self.SerialDevice.write_wfb_lf(bytes(text + '\n', 'utf-8')), 'utf-8')
        return text_out

    def start_serial(self):
        """Open the serial connection to HMP40X0"""
        self.SerialDevice.setup_usb()

        self.SerialDevice.open()
        self.SerialActive = True

        self.__write_to_dev("SYST:MIX")

    def close_serial(self):
        """Closing the serial connection to HMP40X0"""
        self.SerialDevice.close()
        self.SerialActive = False

    def do_reset(self) -> None:
        """Reset the HMP40X0"""
        self.__write_to_dev("*RST")
        sleep(5)
        self.do_beep()

    def do_check_idn(self) -> None:
        """Checking the IDN of HMP40X0"""
        # returned "Rohde&Schwarz,<device type>,<part number>/<serial number>,<firmware version>"
        self.ID_Device = self.__read_from_dev("*IDN?")
        print(self.ID_Device)

    def do_beep(self) -> None:
        """Doing a single beep on device"""
        for ite in range(0, 1):
            self.__write_to_dev("SYST:BEEP")
            sleep(2)

    def ch_read_parameter(self, sel_ch: int) -> [float, float]:
        """Read sense parameter from HMP40X0"""
        text = []
        if sel_ch == 0:
            text.append(self.SetCH1.sel_ch())
            text.append(self.SetCH1.read_ch_voltage())
            text.append(self.SetCH1.read_ch_current())
        if sel_ch == 1:
            text.append(self.SetCH2.sel_ch())
            text.append(self.SetCH2.read_ch_voltage())
            text.append(self.SetCH2.read_ch_current())
        if sel_ch == 2:
            text.append(self.SetCH3.sel_ch())
            text.append(self.SetCH3.read_ch_voltage())
            text.append(self.SetCH3.read_ch_current())
        if sel_ch == 3:
            text.append(self.SetCH4.sel_ch())
            text.append(self.SetCH4.read_ch_voltage())
            text.append(self.SetCH4.read_ch_current())

        # Getting the information
        self.__write_to_dev(text[0])
        volt = float(self.__read_from_dev(text[1]))
        curr = float(self.__read_from_dev(text[2]))

        # Right-back information
        if sel_ch == 0:
            self.SetCH1.set_sense_parameter(volt, curr)
        if sel_ch == 1:
            self.SetCH2.set_sense_parameter(volt, curr)
        if sel_ch == 2:
            self.SetCH3.set_sense_parameter(volt, curr)
        if sel_ch == 3:
            self.SetCH4.set_sense_parameter(volt, curr)

        return volt, curr

    def output_activate(self) -> None:
        """Activate the already setted channels of HMP40X0"""
        self.__write_to_dev("OUTP:GEN ON")
        self.do_beep()

    def output_deactivate(self) -> None:
        """Deactivate all channels of HMP40X0"""
        self.__write_to_dev("OUTP:GEN OFF")

        for idx in range(0, self.__NoChannels):
            text = "INST OUT " + str(idx + 1)
            self.__write_to_dev(text)
            self.__write_to_dev("OUTP OFF")
        self.do_beep()

    def afg_start(self) -> None:
        """Starting the Arbitrary Generator on the configured channel"""
        text = str()
        if self.SetCH1.ChannelUsedAFG:
            text = self.SetCH1.set_arbitrary_start()
        elif self.SetCH2.ChannelUsedAFG:
            text = self.SetCH2.set_arbitrary_start()
        elif self.SetCH3.ChannelUsedAFG:
            text = self.SetCH3.set_arbitrary_start()
        elif self.SetCH4.ChannelUsedAFG:
            text = self.SetCH4.set_arbitrary_start()

        self.__write_to_dev(text)

    def afg_stop(self) -> None:
        """Stopping the Arbitrary Generator on the configured channel"""
        text = str()
        if self.SetCH1.ChannelUsedAFG:
            text = self.SetCH1.set_arbitrary_stop()
        elif self.SetCH2.ChannelUsedAFG:
            text = self.SetCH2.set_arbitrary_stop()
        elif self.SetCH3.ChannelUsedAFG:
            text = self.SetCH3.set_arbitrary_stop()
        elif self.SetCH4.ChannelUsedAFG:
            text = self.SetCH4.set_arbitrary_stop()

        self.__write_to_dev(text)

    def ch_set_parameter(self, sel_ch: int, volt: float, cur: float) -> None:
        """Set parameters I/V of each channel"""
        text = []
        if sel_ch == 0:
            text.append(self.SetCH1.sel_ch())
            text.append(self.SetCH1.set_ch_voltage(volt))
            text.append(self.SetCH1.set_ch_current_limit(cur))
            text.append(self.SetCH1.set_ch_output())
        elif sel_ch == 1:
            text.append(self.SetCH2.sel_ch())
            text.append(self.SetCH2.set_ch_voltage(volt))
            text.append(self.SetCH2.set_ch_current_limit(cur))
            text.append(self.SetCH2.set_ch_output())
        elif sel_ch == 2:
            text.append(self.SetCH3.sel_ch())
            text.append(self.SetCH3.set_ch_voltage(volt))
            text.append(self.SetCH3.set_ch_current_limit(cur))
            text.append(self.SetCH3.set_ch_output())
        elif sel_ch == 3:
            text.append(self.SetCH4.sel_ch())
            text.append(self.SetCH4.set_ch_voltage(volt))
            text.append(self.SetCH4.set_ch_current_limit(cur))
            text.append(self.SetCH4.set_ch_output())

        # Configuring the channels
        for idx in range(0, 4):
            self.__write_to_dev(text[idx])

    def afg_set_waveform(self, sel_ch: int, voltage: np.ndarray, current: np.ndarray, time: np.ndarray, num_cycles=0):
        """Set arbitrary waveform of the selected channel (max. 128 points)"""
        text = []
        if sel_ch == 0:
            text.append(self.SetCH1.set_arbitrary_waveform(voltage, current, time))
            text.append(self.SetCH1.set_arbitrary_repetition(num_cycles))
            text.append(self.SetCH1.set_arbitrary_transfer())
        if sel_ch == 1:
            text.append(self.SetCH2.set_arbitrary_waveform(voltage, current, time))
            text.append(self.SetCH2.set_arbitrary_repetition(num_cycles))
            text.append(self.SetCH2.set_arbitrary_transfer())
        if sel_ch == 2:
            text.append(self.SetCH3.set_arbitrary_waveform(voltage, current, time))
            text.append(self.SetCH3.set_arbitrary_repetition(num_cycles))
            text.append(self.SetCH3.set_arbitrary_transfer())
        if sel_ch == 3:
            text.append(self.SetCH4.set_arbitrary_waveform(voltage, current, time))
            text.append(self.SetCH4.set_arbitrary_repetition(num_cycles))
            text.append(self.SetCH4.set_arbitrary_transfer())

        # Configuring the channels
        for idx in range(0, 3):
            self.__write_to_dev(text[idx])



class ChannelInfoHMP40X0:
    def __init__(self, ch_no: int):
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
        return "INST OUT" + str(self.ChannelNumber)

    def set_ch_voltage(self, voltage_value: float) -> str:
        self.VoltageSet = voltage_value
        return "VOLT " + str(voltage_value)

    def set_ch_current_limit(self, current_value: float) -> str:
        self.CurrentLimit = current_value
        return "CURR " + str(current_value)

    def get_ch_parameter(self) -> str:
        return "APPL?"

    def set_ch_output(self) -> str:
        return "OUTP:SEL 1"

    def read_ch_voltage(self) -> str:
        return "MEAS:VOLT?"

    def read_ch_current(self) -> str:
        return "MEAS:CURR?"

    def set_sense_parameter(self, voltage: float, current: float) -> None:
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
        return "ARB:TRAN " + str(self.ChannelNumber)

    def set_arbitrary_start(self) -> str:
        return "ARB:START " + str(self.ChannelNumber)

    def set_arbitrary_stop(self) -> str:
        return "ARB:STOP " + str(self.ChannelNumber)
