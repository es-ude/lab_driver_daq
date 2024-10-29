from time import sleep
import pyvisa


def scan_instruments(do_print=True) -> list:
    """Scanning the VISA bus for instruments"""
    rm = pyvisa.ResourceManager()
    obj_inst = rm.list_resources()

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


class DriverDMM6500:
    """Class for handling the Keithley Digital Multimeter 6500 in Python"""
    SerialDevice: pyvisa.Resource
    SerialActive = False
    _device_name_chck = "DMM6500"

    def __init__(self):
        pass

    def __write_to_dev(self, order: str) -> None:
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        text_out = self.SerialDevice.query(order)
        return text_out

    def __init_dev(self, do_reset=True, do_beep=True):
        """"""
        self.__write_to_dev("*LANG SCPI")
        if self.SerialActive:
            if do_reset:
                self.do_reset()
            if do_beep:
                self.do_beep()
            print(f"Right device is selected with: {self.get_id(False)}")
        else:
            print("Not right selected device. Please check!")

    def __do_check_idn(self) -> None:
        """Checking the IDN"""
        id_back = self.get_id(False)
        self.SerialActive = self._device_name_chck in id_back

    def serial_start_known_target(self, resource_name: str, do_reset=False) -> None:
        """Open the serial connection to device directly"""
        rm = pyvisa.ResourceManager()
        self.SerialDevice = rm.open_resource(resource_name)

        self.__do_check_idn()
        self.__init_dev(do_reset)

    def serial_start(self, do_reset=False, do_beep=True) -> None:
        """Open the serial connection to device"""
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
        self.__init_dev(do_reset, do_beep)

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

    def do_reset(self) -> None:
        """Reset the device"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev("*RST")
            sleep(1)

    def do_beep(self) -> None:
        """Doing a beep signal"""
        time_sleep = 0.5
        self.__write_to_dev(f':SYST:BEEP 300, {time_sleep}')
        sleep(time_sleep * 1.1)

    def set_measurement_mode(self, mode: int) -> None:
        """Setting the measurement mode"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev(':SENS:FUNC "VOLT:DC"')

    def read_value(self) -> float:
        """Reading value from display"""
        if not self.SerialActive:
            print("... not done due to wrong device")
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

    def get_resistance(self):
        """Get resistance reading
        Args:
            N/A
        Returns:
            Resistance in Ohms
        """
        return float(self.__read_from_dev(":MEAS:RES?"))

    # NOTE: manual page 498
    def __set_measurement_range(self, function: str, polarity: str, range: float):
        """Wrapper for changing ranges
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

        if polarity in available_ranges and function in available_ranges[polarity]:
            ranges = available_ranges[polarity][function]
        else:
            print("Changing measurement range failed. Check polarity. Check function.")
            return True

        for x in ranges:
            if self.__float_eq(float(x), range):
                self.__write_to_dev(f":SENS:{function}:{polarity}:RANG {x}")
                return False

        print("Changing measurement range failed. Check range selection.")
        return True

    def set_resistance_range(self, range: float | str) -> bool:
        from math import log10
        if range == "AUTO":
            self.__write_to_dev(":SENS:RES:RANG:AUTO ON")
            return False
        else:
            range = log10(range)
            if range == int(range) and 0 <= range <= 8:
                self.__write_to_dev(f":SENS:RES:RANG {range}")
                return False
            return True

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
            return self.__set_measurement_range("VOLT", polarity, range)

    def set_current_range(self, range: float | str, polarity: str = "DC") -> bool:
        """Set measurement range of current
        Args:
            range: Available ranges are 0.00001, 0.0001, 0.001, 0.01, 0.1, 1 and 3 Amps
                in DC mode and 0.001, 0.01, 0.1, 1 and 3 Amps in AC mode or just "AUTO"
            polarity: "DC" or "AC", default is "DC"
        Returns:
            True on failure
        """
        if range == "AUTO":
            self.__write_to_dev(":SENS:CURR:RANG:AUTO ON")
            return False
        else:
            return self.__set_measurement_range("CURR", polarity, range)


def main():
    print("Testing device Keithley DMM6500")

    # scan_instruments()
    dev = DriverDMM6500()
    dev.serial_start(do_beep=False)
    dev.do_reset()

    sleep(1)
    for idx in range(0, 5):
        print(dev.read_value())
        sleep(0.5)

    dev.serial_close()


if __name__ == "__main__":
    main()
