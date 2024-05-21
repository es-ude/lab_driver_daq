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

    def __init_dev(self, do_reset=True):
        """"""
        self.__write_to_dev("*LANG SCPI")
        if self.SerialActive:
            if do_reset:
                self.do_reset()
            self.do_beep()
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

    def serial_start_known_target(self, resource_name: str, do_reset=False) -> None:
        """Open the serial connection to device directly"""
        rm = pyvisa.ResourceManager()
        self.SerialDevice = rm.open_resource(resource_name)

        self.__do_check_idn()
        self.__init_dev(do_reset)

    def serial_start(self, do_reset=False) -> None:
        """Open the serial connection to device"""
        list_dev = scan_instruments()
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


if __name__ == "__main__":
    print("Testing device Keithley DMM6500")

    # scan_instruments()
    dev = DriverDMM6500()
    dev.start_serial()
    dev.do_reset()
    #dev.set_measurement_mode(0)
    sleep(1)
    for idx in range(0, 5):
        print(dev.read_value())
        sleep(0.5)
