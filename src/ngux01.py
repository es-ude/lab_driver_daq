import numpy as np
from time import sleep
import pyvisa


def scan_instruments(get_id=False) -> None:
    """Scanning the VISA bus for instruments"""
    rm = pyvisa.ResourceManager()
    print(f"\nAvailable VISA driver: {rm}")
    print("\nAvailable devices")
    print("--------------------------------------")
    obj_inst = rm.list_resources()
    for idx, inst_name in enumerate(obj_inst):
        if get_id:
            inst0 = rm.open_resource(inst_name)
            id = " --> " + inst0.query("*IDN?")
            for ite in range(4):
                inst0.write("SYST:BEEP")
                sleep(0.5)
            inst0.close()
        else:
            id = ""
        print(f"{idx}: {inst_name}{id}")


class NGUX01:
    """Class for handling the Rohde and Schwarz Sourcemeter NGUX01 in Python"""
    SerialDevice: pyvisa.Resource
    SerialActive = False
    _volt_range = [-20.0, 20.0]
    _curr_range = [-0.1, 0.1]

    def __init__(self):
        pass

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
        """Closing the serial connection"""
        self.SerialDevice.close()
        self.SerialActive = False

    def do_reset(self) -> None:
        """Reset the device"""
        self.__write_to_dev("*RST")
        sleep(5)
        self.do_beep()

    def do_check_idn(self) -> None:
        """Checking the IDN"""
        self.ID_Device = self.__read_from_dev("*IDN?")
        print(self.ID_Device)

    def do_beep(self) -> None:
        """Doing a single beep on device"""
        for ite in range(0, 1):
            self.__write_to_dev("SYST:BEEP")
            sleep(2)

    def do_get_system_runtime(self) -> None:
        """Getting the actual runtime of device in seconds"""
        runtime = int(self.__read_from_dev('SYST:UPT?'))
        print(f"Actual runtime: {runtime/60:.3f} sec")

    def do_get_system_time(self) -> None:
        """Getting the actual system time of device """
        time = self.__read_from_dev('SYST:TIME?').split(',')
        print(f"System time: {int(time[0]):02d}:{int(time[1]):02d},{int(time[2]):02d}")

    def set_voltage(self, val: float) -> None:
        """Setting the channel voltage value"""
        val_set = val if val > self._volt_range[0] else self._volt_range[0]
        val_set = val_set if val <= self._volt_range[1] else self._volt_range[1]
        self.__write_to_dev(f"VOLT {val_set:.4f}")

    def set_current(self, val: float) -> None:
        """Setting the channel voltage value"""
        val_set = val if val > self._curr_range[0] else self._curr_range[0]
        val_set = val_set if val <= self._curr_range[1] else self._curr_range[1]
        self.__write_to_dev(f"CURR {val_set:.4f}")

    def set_current_limits(self, val_min: float, val_max: float) -> None:
        """Setting the current limitations value"""
        self.__write_to_dev(f"CURR:ALIM {val_max:.4f}")
        sleep(0.5)
        self.__write_to_dev(f"CURR:ALIM LOW {val_min:.4f}")
        sleep(0.5)

    def set_output_impedance(self, resistance: float) -> None:
        """Setting the output impedance of device"""
        self.__write_to_dev(f"OUTP:IMP {resistance:.5f}")

    def set_output_mode(self, mode=0) -> None:
        """Setting the output mode [0: Auto, 1: Sink, 2: Source]"""
        str_out = 'AUTO'
        match mode:
            case 0:
                str_out = 'AUTO'
            case 1:
                str_out = 'SINK'
            case 2:
                str_out = 'SOUR'
        self.__write_to_dev(f"OUTP:MODE {str_out}")

    def output_activate(self, use_fast_output=False) -> None:
        """Activating the output"""
        self.__write_to_dev(f"OUTP:FTR {1 if use_fast_output else 0}")
        sleep(0.5)
        self.__write_to_dev(f"OUTP:SEL 1")
        sleep(0.5)
        self.__write_to_dev(f"OUTP:GEN 1")
        sleep(0.5)

    def output_deactivated(self) -> None:
        """Deactivating the output"""
        self.__write_to_dev(f"OUTP:SEL 0")
        sleep(0.5)
        self.__write_to_dev(f"OUTP:GEN 0")
        sleep(0.5)

    def get_measurement_voltage(self) -> None:
        """Reading the voltage"""
        val = float(self.__read_from_dev("MEAS:SCAL:VOLT?"))
        print(f"... meas. voltage: {val:.6f} V")

    def get_measurement_current(self) -> None:
        """Reading the current"""
        val = float(self.__read_from_dev("MEAS:SCAL:CURR?"))
        print(f"... meas. voltage: {val:.6f} V")

    def get_measurement_power(self) -> None:
        """Reading the current"""
        val = float(self.__read_from_dev("MEAS:SCAL:PWR?"))
        print(f"... meas. voltage: {val:.6f} V")

    def get_measurement_energy(self) -> None:
        """Reading the current"""
        val = float(self.__read_from_dev("MEAS:SCAL:ENER?"))
        print(f"... meas. voltage: {val:.6f} V")


if __name__ == "__main__":
    scan_instruments()
