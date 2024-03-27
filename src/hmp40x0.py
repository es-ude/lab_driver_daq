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


class DriverHMP40X0:
    """Class for Remote Controlling the Power Supply R&S HMP40X0 via USB"""
    SerialDevice: pyvisa.Resource

    def __init__(self, num_ch=4):
        """Initizialization of Power Supply Device with BAUD=9600(com_name, num_of_channels)"""
        self.__NoChannels = num_ch
        self.SerialActive = False
        self.ChannelUsed = [False, False, False, False]
        self.ID_Device = str()
        self.SetCH = [ChannelInfoHMP40X0(idx) for idx in range(num_ch)]

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
        for idx in range(self.__NoChannels):
            if self.SetCH[idx].ChannelUsedAFG:
                text = self.SetCH[idx].set_arbitrary_start()
        self.__write_to_dev(text)

    def afg_stop(self) -> None:
        """Stopping the Arbitrary Generator on the configured channel"""
        text = str()
        for idx in range(self.__NoChannels):
            if self.SetCH[idx].ChannelUsedAFG:
                text = self.SetCH[idx].set_arbitrary_stop()
        self.__write_to_dev(text)

    def ch_set_parameter(self, sel_ch: int, volt: float, cur: float) -> None:
        """Set parameters I/V of each channel"""
        text = []
        text.append(self.SetCH[sel_ch].sel_ch())
        text.append(self.SetCH[sel_ch].set_ch_voltage(volt))
        text.append(self.SetCH[sel_ch].set_ch_current_limit(cur))
        text.append(self.SetCH[sel_ch].set_ch_output())

        # Configuring the channels
        for string in text:
            self.__write_to_dev(string)

    def afg_set_waveform(self, sel_ch: int, voltage: np.ndarray, current: np.ndarray, time: np.ndarray, num_cycles=0):
        """Set arbitrary waveform of the selected channel (max. 128 points)"""
        text = []
        text.append(self.SetCH[sel_ch].set_arbitrary_waveform(voltage, current, time))
        text.append(self.SetCH[sel_ch].set_arbitrary_repetition(num_cycles))
        text.append(self.SetCH[sel_ch].set_arbitrary_transfer())

        # Configuring the channels
        for string in text:
            self.__write_to_dev(string)


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


if __name__ == "__main__":
    scan_instruments()

    sleep(1)
    inst_dev = DriverHMP40X0()
    inst_dev.start_serial('ASRL4::INSTR')
    inst_dev.do_beep()
    inst_dev.ch_set_parameter(0, 1.6, 10e-3)
