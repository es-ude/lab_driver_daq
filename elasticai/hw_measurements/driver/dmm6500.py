from time import sleep
from logging import getLogger, Logger
import pyvisa
from serial.tools import list_ports
from elasticai.hw_measurements.scan_instruments import scan_instruments


class DriverDMM6500:
    SerialDevice: pyvisa.Resource
    SerialActive = False
    _device_name_chck = "DMM6500"
    _logger: Logger
    _usb_vid = 0x05e6
    _usb_pid = 0x6500

    def __init__(self):
        """Class for handling the Keithley Digital Multimeter 6500 in Python"""
        self._logger = getLogger(__name__)

    def __write_to_dev(self, order: str) -> None:
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        num_stop = 100
        num_trials = 0
        while (num_trials < num_stop):
            try:
                text_out = self.SerialDevice.query(order)
                num_trials = num_stop
            except:
                sleep(0.1)
                num_trials += 1
        return text_out.strip()

    def __init_dev(self, do_reset: bool=True, do_beep: bool=True) -> None:
        """Function for initialisation of DAQ device
        :param do_reset:    Reset the DAQ device
        :param do_beep:     Do a beep on DAQ device after init done
        :return: None
        """
        self.__write_to_dev("*LANG SCPI")
        if self.SerialActive:
            if do_reset:
                self.do_reset()
            if do_beep:
                self.do_beep()
            self._logger.debug(f"Right device is selected with: {self.get_id()}")
        else:
            self._logger.debug("Not right selected device. Please check!")

    def __do_check_idn(self) -> None:
        """Checking the IDN"""
        id_back = self.get_id()
        self.SerialActive = self._device_name_chck in id_back

    def serial_start_known_target(self, resource_name: str, do_reset: bool=False) -> None:
        """Open the serial connection to device directly"""
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

    def serial_start(self, do_reset: bool=False, do_beep: bool=True) -> None:
        """Open the serial connection to device
        :param do_reset:    Reset the DAQ device
        :param do_beep:     Do a beep on DAQ device after init done
        :return:            None
        """
        list_dev = scan_instruments(self)
        rm = pyvisa.ResourceManager("@py")

        # --- Checking if device address is right
        for inst_name in list_dev:
            self.SerialDevice = rm.open_resource(inst_name)
            self.__do_check_idn()
            if self.SerialActive:
                break
            else:
                self.serial_close()

        # --- Init of device
        self.__init_dev(do_reset, do_beep)

    def serial_close(self) -> None:
        """Closing the serial connection"""
        self.SerialDevice.close()
        self.SerialActive = False

    def get_id(self) -> str:
        """Getting the device ID"""
        id = self.__read_from_dev("*IDN?")
        self._logger.debug(f"Device ID: {id}")
        return id

    def do_reset(self) -> None:
        """Reset the device"""
        if not self.SerialActive:
            self._logger.debug("... RST not done due to wrong device")
        else:
            self.__write_to_dev("*RST")
            sleep(1)

    def do_beep(self) -> None:
        """Doing a beep signal"""
        time_sleep = 0.5
        self.__write_to_dev(f':SYST:BEEP 300, {time_sleep}')
        sleep(time_sleep * 1.1)

    def set_measurement_mode(self, mode: str, polarity: str = "") -> None:
        """Set the measurement mode
        Args:
            mode: "VOLT", "CURR", "RES" or "FRES"
            polarity: "DC" or "AC" applicable for current and voltage, else ""
        Returns:
            None
        """
        if polarity:
            self.__write_to_dev(f':SENS:FUNC "{mode}:{polarity}"')
        else:
            self.__write_to_dev(f':SENS:FUNC "{mode}"')

    def read_value(self) -> float:
        """Reading value from display"""
        if not self.SerialActive:
            self._logger.debug("... reading not done due to wrong device")
        else:
            return float(self.__read_from_dev(":READ?"))

    def __float_eq(self, x, y, epsilon=0.00000001):
        return abs(x - y) < epsilon

    def get_voltage(self):
        """Get voltage reading
        Args:
            N/A
        Returns:
            Voltage in Volts
        """
        return float(self.__read_from_dev(":MEAS:VOLT?"))

    def get_current(self):
        """Get current reading
        Args:
            N/A
        Returns:
            Current in Ampere
        """
        return float(self.__read_from_dev(":MEAS:CURR?"))

    def get_2wire_resistance(self):
        """Get 2-wire resistance reading
        Args:
            N/A
        Returns:
            2-wire resistance in Ohms
        """
        return float(self.__read_from_dev(":MEAS:RES?"))

    def get_4wire_resistance(self):
        """Get 4-wire resistance reading
        Args:
            N/A
        Returns:
            4-wire resistance in Ohms
        """
        return float(self.__read_from_dev(":MEAS:FRES?"))

    # NOTE: manual page 498
    def __set_volt_curr_range(self, function: str, polarity: str, range: float):
        """Wrapper for changing ranges of voltage or current readings
        Args:
            function: "VOLT" or "CURR"
            polarity: "AC" or "DC"
            range: One of the available ranges for that selection
        Returns:
            True on failure
        """
        available_ranges = {
            "DC" : {
                "VOLT": ["1e-1", "1", "10", "100", "1000"],
                "CURR": ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1", "3"]
            },
            "AC" : {
                "VOLT": ["1e-1", "1", "10", "100", "750"],
                "CURR": ["1e-3", "1e-2", "1e-1", "1", "3"]
            }
        }

        if self.__read_from_dev(":ROUT:TERM?") == "REAR":
            available_ranges["DC"]["CURR"].append("10")
            available_ranges["AC"]["CURR"].append("10")

        if polarity in available_ranges and function in available_ranges[polarity]:
            ranges = available_ranges[polarity][function]
        else:
            self._logger.info("Changing measurement range failed. Check polarity. Check function.")
            return True

        for x in ranges:
            if self.__float_eq(float(x), range):
                self.__write_to_dev(f":SENS:{function}:{polarity}:RANG {x}")
                return False

        self._logger.info("Changing measurement range failed. Check range selection.")
        return True

    def set_4wire_offset_compensation(self, compensation: bool | str) -> bool:
        """Enable or disable offset compensation for 4-wire resistance
        Args:
            compensation: True to enable, False to disable, "AUTO" to enable automatically
        Returns:
            True on failure; invalid argument
        """
        if compensation == "AUTO":
            self.__write_to_dev(":SENS:FRES:OCOM AUTO")
        elif type(compensation) == bool:
            self.__write_to_dev(f":SENS:FRES:OCOM {int(compensation)}")
        else:
            self._logger.debug("Setting 4-wire resistance offset compensation failed. Check argument.")
            return True
        return False

    def __set_resistance_range(self, range: int | str, type: int) -> bool:
        """Wrapper for changing ranges of resistance readings
        Args:
            range: One of the available ranges for that type of resistance reading or "AUTO".
                All ranges are powers of 10.
                2-wire ranges are 10^1 to 10^8.
                4-wire ranges are 10^0 to 10^8 if offset compensation is off or auto.
                4-wire ranges are 10^0 to 10^4 if offset compensation is on.
            type: 2 or 4 to change resistance range of 2-wire of 4-wire resistance.
        Returns:
            True on failure
        """
        if range == "AUTO":
            if type not in (2,4):
                self._logger.info(f"Only 2-wire and 4-wire resistance types are supported. You selected {type}.")
                return True
            self.__write_to_dev(":SENS:#RES:RANG:AUTO ON".replace('#', 'F' if type == 4 else ''))
            return False

        try:
            from math import log10
            power = log10(range)
            if power != int(power):
                self._logger.info("Range argument must be power of 10.")
                return True
        except:
            self._logger.info(f"Mathematical error during computation of log10 of range argument: {range}.")
            return True

        if type == 2:
            if 1 <= power <= 8:
                self.__write_to_dev(f":SENS:RES:RANG {range}")
                return False
        elif type == 4:
            offset_comp = self.__read_from_dev(":SENS:FRES:OCOM?")
            valid = offset_comp == "ON" and 0 <= power <= 4
            valid |= offset_comp in ("OFF", "AUTO") and 0 <= power <= 8
            if valid:
                self.__write_to_dev(f":SENS:FRES:RANG {range}")
                return False
        else:
            self._logger.info(f"Only 2-wire and 4-wire resistance types are supported. You selected {type}.")
            return True
        self._logger.info(f"Range argument is out of supported range: {range}")
        return True

    def set_2wire_resistance_range(self, range: int | str) -> bool:
        """Set measurement range of 2-wire resistance
        Args:
            range: Power of 10 from 10^1 to 10^8 or "AUTO"
        Returns:
            True on failure
        """
        return self.__set_resistance_range(range, 2)

    def set_4wire_resistance_range(self, range: int | str) -> bool:
        """Set measurement range of 4-wire resistance
        Args:
            range: Power of 10 from 10^0 to 10^8 if offset compensation is off or "AUTO",
                else 10^0 to 10^4 if offset compensation is on.
        Returns:
            True on failure
        """
        return self.__set_resistance_range(range, 4)

    def set_voltage_range(self, range: float | str, polarity: str = "DC") -> bool:
        """Set measurement range of voltage
        Args:
            range: Available ranges are 0.1, 1, 10, 100 and 1000 Volts in DC mode
                and 0.1, 1, 10, 100, 750 in AC mode or just "AUTO"
            polarity: "DC" or "AC", default is "DC"
        Returns:
            True on failure
        """
        if range == "AUTO":
            self.__write_to_dev(":SENS:VOLT:RANG:AUTO ON")
            return False
        else:
            return self.__set_volt_curr_range("VOLT", polarity, range)

    def set_current_range(self, range: float | str, polarity: str = "DC") -> bool:
        """Set measurement range of current
        Args:
            range: Available ranges are 0.00001, 0.0001, 0.001, 0.01, 0.1, 1 and 3 Amps
                in DC mode and 0.001, 0.01, 0.1, 1 and 3 Amps in AC mode or just "AUTO".
                When the rear terminals are used, 10 Amp range is available.
            polarity: "DC" or "AC", default is "DC"
        Returns:
            True on failure
        """
        if range == "AUTO":
            self.__write_to_dev(":SENS:CURR:RANG:AUTO ON")
            return False
        else:
            return self.__set_volt_curr_range("CURR", polarity, range)
