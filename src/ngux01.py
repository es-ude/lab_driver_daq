from time import sleep
import pyvisa


def scan_instruments() -> list:
    """Scanning the VISA bus for instruments"""
    rm = pyvisa.ResourceManager()
    print(f"\nAvailable VISA driver: {rm}")
    print("\nAvailable devices")
    print("--------------------------------------")
    obj_inst = rm.list_resources()
    out_dev_adr = list()
    for idx, inst_name in enumerate(obj_inst):
        print(f"{idx}: {inst_name}")
        out_dev_adr.append(inst_name)
    return out_dev_adr


class NGUX01:
    """Class for handling the Rohde and Schwarz Sourcemeter NGUX01 in Python"""
    SerialDevice: pyvisa.Resource
    SerialActive = False
    _volt_range = [-20.0, 20.0]
    _curr_range = [-0.1, 0.1]
    _device_name_chck = "NGU"

    def __init__(self):
        pass

    def __write_to_dev(self, order: str) -> None:
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        text_out = self.SerialDevice.query(order)
        return text_out

    def __run_code(self, function):
        """"""
        if self.SerialActive:
            return function
        else:
            print("Not right device available. Doing nothing!")

    def __init_dev(self, do_reset=True):
        """"""
        if self.SerialActive:
            if do_reset:
                self.do_reset()
            self.__write_to_dev("SYST:MIX")
            print(f"Right device is selected with: {self.get_id()}")
        else:
            print("Not right selected device. Please check!")

    def __do_check_idn(self, do_print=True) -> None:
        """Checking the IDN"""
        id_back = self.get_id(do_print)
        if self._device_name_chck in id_back:
            self.SerialActive = True
        else:
            if do_print:
                print("... Device not right. Please check other version and restart!")
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
            self.__do_check_idn(False)
            if self.SerialActive:
                break

        self.__init_dev(do_reset)

    def close_serial(self) -> None:
        """Closing the serial connection"""
        self.SerialDevice.close()
        self.SerialActive = False

    def do_reset(self) -> None:
        """Reset the device"""
        self.__write_to_dev("*RST")
        sleep(5)
        self.do_beep()

    def get_id(self, do_print=True) -> str:
        """Getting the device ID"""
        id = self.__read_from_dev("*IDN?")
        if do_print:
            print(id)
        return id

    def do_beep(self, num_iterations=1) -> None:
        """Doing a single beep on device"""
        for ite in range(0, num_iterations):
            self.__write_to_dev("SYST:BEEP")
            sleep(1)

    def do_get_system_runtime(self) -> None:
        """Getting the actual runtime of device in seconds"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            runtime = int(self.__read_from_dev('SYST:UPT?'))
            print(f"Actual runtime: {runtime / 60:.3f} sec")

    def do_get_system_time(self) -> None:
        """Getting the actual system time of device """
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            time = self.__read_from_dev('SYST:TIME?').split(',')
            print(f"System time: {int(time[0]):02d}:{int(time[1]):02d},{int(time[2]):02d}")

    def set_voltage(self, val: float) -> None:
        """Setting the channel voltage value"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            val_set = val if val > self._volt_range[0] else self._volt_range[0]
            val_set = val_set if val <= self._volt_range[1] else self._volt_range[1]
            self.__write_to_dev(f"VOLT {val_set:.4f}")

    def set_current(self, val: float) -> None:
        """Setting the channel voltage value"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            val_set = val if val > self._curr_range[0] else self._curr_range[0]
            val_set = val_set if val <= self._curr_range[1] else self._curr_range[1]
            self.__write_to_dev(f"CURR {val_set:.4f}")

    def set_current_limits(self, val_min: float, val_max: float) -> None:
        """Setting the current limitations value"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev(f"CURR:ALIM {val_max:.4f}")
            sleep(0.5)
            self.__write_to_dev(f"CURR:ALIM LOW {val_min:.4f}")
            sleep(0.5)

    def set_output_impedance(self, resistance: float) -> None:
        """Setting the output impedance of device"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev(f"OUTP:IMP {resistance:.5f}")

    def set_output_mode(self, mode=0) -> None:
        """Setting the output mode [0: Auto, 1: Sink, 2: Source]"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            if mode == 0:
                str_out = 'AUTO'
            elif mode == 1:
                str_out = 'SINK'
            else:
                str_out = 'SOUR'
            self.__write_to_dev(f"OUTP:MODE {str_out}")

    def output_activate(self, use_fast_output=False) -> None:
        """Activating the output"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev(f"OUTP:FTR {1 if use_fast_output else 0}")
            sleep(0.5)
            self.__write_to_dev(f"OUTP:SEL 1")
            sleep(0.5)
            self.__write_to_dev(f"OUTP:GEN 1")
            sleep(0.5)

    def output_deactivated(self) -> None:
        """Deactivating the output"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev(f"OUTP:SEL 0")
            sleep(0.5)
            self.__write_to_dev(f"OUTP:GEN 0")
            sleep(0.5)

    def get_measurement_voltage(self, do_print=True) -> float:
        """Reading the voltage"""
        if not self.SerialActive:
            print("... not done due to wrong device")
            return 0.0
        else:
            val = float(self.__read_from_dev("MEAS:SCAL:VOLT?"))
            if do_print:
                print(f"... meas. voltage: {val:.6f} V")
            return val

    def get_measurement_current(self, do_print=True) -> float:
        """Reading the current"""
        if not self.SerialActive:
            print("... not done due to wrong device")
            return 0.0
        else:
            val = float(self.__read_from_dev("MEAS:SCAL:CURR?"))
            if do_print:
                print(f"... meas. current: {1e3 * val:.6f} mA")
            return val

    def get_measurement_power(self, do_print=True) -> float:
        """Reading the power"""
        if not self.SerialActive:
            print("... not done due to wrong device")
            return 0.0
        else:
            val = float(self.__read_from_dev("MEAS:SCAL:PWR?"))
            if do_print:
                print(f"... meas. power: {1e3 * val:.6f} mW")
            return val

    def get_measurement_energy(self, do_print=True) -> float:
        """Reading the energy"""
        if not self.SerialActive:
            print("... not done due to wrong device")
            return 0.0
        else:
            val = float(self.__read_from_dev("MEAS:SCAL:ENER?"))
            if do_print:
                print(f"... meas. energy: {1e3 * val:.6f} mWh")
            return val


if __name__ == "__main__":
    scan_instruments()

    dev = NGUX01()
    dev.start_serial()
