import pyvisa
from platform import system
from time import sleep, strftime, time_ns
from logging import getLogger, Logger
from serial.tools import list_ports
from lab_driver.scan_instruments import scan_instruments


class DriverNGUX01:
    SerialDevice: pyvisa.Resource
    SerialActive = False
    _logger: Logger
    _usb_vid = 0x0aad
    _usb_pid = 0x0197    # for some reason NGU and MXO share the same PID

    _volt_range = [-20.0, 20.0]
    _curr_range = [-0.1, 0.1]
    _device_name_chck = "NGU"
    _last_usb_state: bool   # True = connected, False = disconnected
    _fastlog_finish_timestamp: int = 0

    def __init__(self):
        """Class for handling the Rohde and Schwarz Sourcemeter NGUX01 in Python"""
        self._logger = getLogger(__name__)

    def __write_to_dev(self, order: str) -> None:
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        text_out = self.SerialDevice.query(order)
        return text_out

    def __init_dev(self, do_reset=True):
        """"""
        if self.SerialActive:
            if do_reset:
                self.do_reset()
            self.__write_to_dev("SYST:MIX")
            self.__write_to_dev(strftime("SYST:TIME %H,%M,%S"))
            self._last_usb_state = self.is_usb_connected()
            print(f"Right device is selected with: {self.get_id(False)}")
            self.sync()
        else:
            print("Not right selected device. Please check!")

    def __do_check_idn(self) -> None:
        """Checking the IDN"""
        id_back = self.get_id(False)
        if self._device_name_chck in id_back:
            self.SerialActive = True
        else:
            self.SerialActive = False

    def __float_eq(self, x, y, epsilon=0.00000001):
        return abs(x - y) < epsilon

    def sync(self, timeout = 86400000) -> None:
        """Wait until all queued commands have been processed
        Args:
            timeout: timeout in milliseconds, VISA exception thrown on timeout, default 1 day
        Returns:
            None
        """
        backup_timeout = self.SerialDevice.timeout
        self.SerialDevice.timeout = timeout
        self.__write_to_dev("*WAI")
        self.SerialDevice.timeout = backup_timeout

    def serial_open_known_target(self, resource_name: str, do_reset: bool=False) -> None:
        """Open the serial connection to device"""
        rm = pyvisa.ResourceManager()
        self.SerialDevice = rm.open_resource(resource_name)

        try:
            self.__do_check_idn()
        except:
            raise RuntimeError('Device not connected or USB class of NGU is not TMC (set per menu -> Interfaces -> USB class)')
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

    def serial_start(self, do_reset: bool=False) -> None:
        """Open the serial connection to device"""
        list_dev = scan_instruments(self)
        if system() == "Linux":
            rm = pyvisa.ResourceManager("/usr/lib/librsvisa.so@ivi")
        else:
            rm = pyvisa.ResourceManager()

        for inst_name in list_dev:
            self.SerialDevice = rm.open_resource(inst_name)
            try:
                self.__do_check_idn()
            except:
                raise RuntimeError(
                    'Device not connected or USB class of NGU is not TMC (set per menu -> Interfaces -> USB class)')

            if self.SerialActive:
                break
            else:
                self.serial_close()

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
                sleep(.3)

    def do_reset(self) -> None:
        """Reset the device"""
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev("*RST")
            sleep(2)
            self.do_beep()

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

    def set_voltage_range(self, range: int) -> bool:
        """Set measurement voltage range
        Args:
            range: voltage range 6 or 20 volts
        Returns:
            True when argument is invalid
        """
        if range in (6,20):
            self.__write_to_dev(f"VOLT:RANG {range}")
            return False
        return True

    def get_voltage_range(self) -> float:
        """Get measurement voltage range
        Returns:
            Range 6 or 20 volts
        """
        return float(self.__read_from_dev("VOLT:RANG?"))

    def set_current_range(self, range: float) -> bool:
        """Set measurement current range
        Args:
            range: current range 2, 0.1 or 0.01
        Returns:
            True when argument is invalid
        """
        available_ranges = ["2", "0.1", "0.01"]
        for x in available_ranges:
            if self.__float_eq(range, float(x)):
                self.__write_to_dev(f"CURR:RANG {x}")
                return False
        return True

    def get_current_range(self) -> float:
        """Get measurement current range
        Returns:
            Range from 0 to 8 Amps
        """
        return float(self.__read_from_dev("CURR:RANG?"))

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
        """Setting the current limitations value
        Args:
            val_min: Lower limit in Amps
            val_max: Upper limit in Amps
        Returns:
            None
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev(f"CURR:ALIM {val_max:.4f}")
            sleep(0.5)
            self.__write_to_dev(f"CURR:ALIM LOW {val_min:.4f}")
            sleep(0.5)

    def set_output_impedance(self, resistance: float) -> None:
        """Setting the output impedance of device
        Args:
            resistance: Impedance in Ohms
        Returns:
            None
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev(f"OUTP:IMP {resistance:.5f}")

    def set_output_mode(self, mode=0) -> None:
        """Setting the output mode
        Args:
            mode: 0 = Auto, 1 = Sink, 2 = Source
        Returns:
            None
        """
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

    def output_activate(self, use_fast_output: bool=False) -> None:
        """Activating the output
        Args:
            use_fast_output: (De-)Activate fast transient response
        Returns:
            None
        """
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
        """Deactivating the output
        Returns:
            None
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev(f"OUTP:SEL 0")
            sleep(0.5)
            self.__write_to_dev(f"OUTP:GEN 0")
            sleep(0.5)

    def get_measurement_voltage(self, do_print: bool=False) -> float:
        """Reading the voltage
        Args:
            do_print: Also print the voltage value to stdout
        Returns:
            The voltage value
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
            return 0.0
        else:
            val = float(self.__read_from_dev("MEAS:SCAL:VOLT?"))
            if do_print:
                print(f"... meas. voltage: {val:.6f} V")
            return val

    def get_measurement_current(self, do_print: bool=False) -> float:
        """Reading the current
        Args:
            do_print: Also print the current value to stdout
        Returns:
            The current value
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
            return 0.0
        else:
            val = float(self.__read_from_dev("MEAS:SCAL:CURR?"))
            if do_print:
                print(f"... meas. current: {1e3 * val:.6f} mA")
            return val

    def get_measurement_power(self, do_print: bool=False) -> float:
        """Reading the power
        Args:
            do_print: Also print the power value to stdout
        Returns:
            The power value
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
            return 0.0
        else:
            val = float(self.__read_from_dev("MEAS:SCAL:PWR?"))
            if do_print:
                print(f"... meas. power: {1e3 * val:.6f} mW")
            return val

    def get_measurement_energy(self, do_print: bool=False) -> float:
        """Reading the energy
        Args:
            do_print: Also print the energy value to stdout
        Returns:
            The energy value
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
            return 0.0
        else:
            val = float(self.__read_from_dev("MEAS:SCAL:ENER?"))
            if do_print:
                print(f"... meas. energy: {1e3 * val:.6f} mWh")
            return val

    def set_fastlog_sample_rate(self, rate: float) -> bool:
        """Set sample rate of FastLog
        Args:
            rate: Either 500, 250, 50, 10, 1 or 0.1 kilosamples per second
        Returns:
            True when argument is invalid
        """
        if self.__float_eq(rate, 0.1):
            self.__write_to_dev("FLOG:SRAT S100")
        elif rate in (1, 10, 50, 250, 500):
            self.__write_to_dev(f"FLOG:SRAT S{rate}k")
        else:
            return True
        return False

    def get_fastlog_sample_rate(self) -> float:
        """Get sample rate of FastLog
        Returns:
            FastLog sample rate in kilosamples per second
        """
        rate = self.__read_from_dev("FLOG:SRAT?").strip()[1:]   # rate has format "S###[k]", get rid of the S
        if rate[-1] == 'k':
            return float(rate[:-1])
        else:
            return 0.1

    def set_fastlog_duration(self, duration: int) -> None:
        """Set duration of a FastLog sample
        Args:
            duration: Sample duration in seconds
        Returns:
            None
        """
        self.__write_to_dev(f"FLOG:FILE:DUR {duration}")

    def get_fastlog_duration(self) -> int:
        """Get duration of a FastLog sample
        Returns:
            Duration in seconds
        """
        return float(self.__read_from_dev("FLOG:FILE:DUR?"))

    def set_fastlog_triggered(self, triggered: bool) -> None:
        """Set whether FastLog is started by a trigger event
        Args:
            triggered: True to receive triggers, False to ignore
        Returns:
            None
        """
        self.__write_to_dev(f"FLOG:TRIG {int(triggered)}")

    def get_fastlog_triggered(self) -> bool:
        """Get whether FastLog is started by a trigger event
        Returns:
            True if trigger is activated, False if not
        """
        return bool(int(self.__read_from_dev("FLOG:TRIG?")))

    def do_fastlog(self, duration: int = None, sample_rate: float = None) -> bool:
        """Start a FastLog measurement
        Args:
            duration: Optional; set the duration of the measurement
            sample_rate: Optional; set a sample rate before measuring
        Returns:
            True when arguments are invalid or USB device not detected
        """
        if self.is_usb_disconnected():
            return True
        if sample_rate is not None and self.set_fastlog_sample_rate(sample_rate):
            return True
        if duration is not None:
            self.set_fastlog_duration(duration)
        self.__write_to_dev("FLOG 1")
        self._fastlog_finish_timestamp = time_ns() + self.get_fastlog_duration() * 10**9
        return False

    def abort_fastlog(self) -> None:
        """Abort a running FastLog measurement. Progress is saved.
        Args:
            N/A
        Returns:
            None
        """
        self.__write_to_dev("FLOG 0")
        self._fastlog_finish_timestamp = time_ns()
    
    def is_usb_connected(self) -> bool:
        """EVENT - Check for USB device connection
        Returns:
            True if USB device is detected, False otherwise
        """
        return "USB" in self.test("FLOG:FILE:TPAR?")
    
    def is_usb_disconnected(self) -> bool:
        """EVENT - Convenience function for negation of is_usb_connected, to avoid lambda expression
        Returns:
            True if no USB device is detected, False otherwise
        """
        return not self.is_usb_connected()
    
    def has_usb_switched_state(self) -> bool:
        """EVENT - Check if USB device has been (dis-)connected since the last call of this function
        or since serial connection has been established if called for the first time
        Returns:
            True if USB device has been disconnected when it was connected before or vice versa
        """
        now_state = self.is_usb_connected()
        ret = now_state != self._last_usb_state
        self._last_usb_state = now_state
        return ret
    
    def is_fastlog_running(self) -> bool:
        """EVENT - Check if FastLog measurement is currently running
        Returns:
            True if FastLog measurement is running, False otherwise
        """
        return time_ns() <= self._fastlog_finish_timestamp and self.is_usb_connected()
    
    def is_fastlog_finished(self) -> bool:
        """EVENT - Convenience function for negation of is_fastlog_running, to avoid lambda expression
        Returns:
            True if FastLog measurement has finished or is not running, False otherwise
        """
        return not self.is_fastlog_running()
    
    def event_handler(self, event, action, *args, **kwargs):
        """Listen for events and execute an action when triggered. The desired event is
        polled for every 100 ms, sleeping in-between, hence this is a blocking function.
        Args:
            event: Some event listener function
            action: Any action to be executed
            *args: Positional arguments passed to action
            **kwargs: Keyword arguments passed to action
        Returns:
            Whatever the action returns
        """
        while not event():
            sleep(0.1)
        return action(*args, **kwargs)
        
    def test(self, cmd):
        if '?' in cmd:
            return self.__read_from_dev(cmd)
        else:
            self.__write_to_dev(cmd)
