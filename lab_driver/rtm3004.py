from mxo4x import *

class DriverRTM3004(DriverMXO4X):
    _device_name_chck = "RTM"

    def __write_to_dev(self, order: str) -> None:
        """Wrapper for executing commands on device
        Args:
            order: command to run on device (may alter device state)
        Returns:
            None
        """
        try:
            self.SerialDevice.write(order)
        except Exception as e:
            self._cmd_stack.append((order, f"FAILED - {e}"))
            raise e
        else:
            self._cmd_stack.append(order)

    def __read_from_dev(self, order: str) -> str:
        """Wrapper for querying data from device
        Args:
            order: command to run on device
        Returns:
            Queried data as a string
        """
        try:
            text_out = ""   # default value needed, else variable may never be assigned!
            text_out = self.SerialDevice.query(order)
        except Exception as e:
            self._cmd_stack.append((order, f"FAILED - {e} - {text_out}"))
            raise e
        else:
            self._cmd_stack.append((order, text_out))
        return text_out

    def get_id(self, do_print=True) -> str:
        """Getting the device ID
        Args:
            do_print: optionally print the device ID to stdout
        Returns:
            Device ID as a string
        """
        # For some reason the ID ends on three null characters, so strip the string 
        id = self.__read_from_dev("*IDN?").strip("\0").strip()
        if do_print:
            print(id)
        return id

    def __init_dev(self, do_reset=True):
        """If the correct device is selected, initialise it and optionally do a reset
        Args:
            do_reset: reset device or not
        Returns:
            None
        """
        if self.SerialActive:
            if do_reset:
                self.do_reset()
            # This command doesn't seem to exist? Windows doesn't throw exceptions on wrong commands,
            # that's why it worked there, but not on Linux
            # self.__write_to_dev("SYST:MIX")   # Instrument error detected: -113,"Undefined header;SYST:MIX"
            print(f"Right device is selected with: {self.get_id(False)}")
            self.sync_device_time()
        else:
            print("Not right selected device. Please check!")

    def __do_check_idn(self) -> None:
        """Checking the IDN"""
        id_back = self.get_id(False)
        self.SerialActive = self._device_name_chck in id_back
        if self.SerialActive:
            self._firmware_version = id_back.split(',')[-1]
    
    def serial_start(self, do_reset=False) -> None:
        """Open the serial connection to device if it is found
        Args:
            do_reset: reset device during initialisation
        Returns:
            None
        """
        if False and platform.system() == "Linux":
            # Resource string for RTM3004
            self.serial_open_known_target("USB0::0x0AAD::0x01D6::113613::INSTR", do_reset)
            return

        list_dev = scan_instruments(do_print=False)
        rm = pyvisa.ResourceManager(self._visa_lib)

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

    def gen_enable(self) -> None:
        """Enable waveform generator
        Returns:
            None
        """
        self.__write_to_dev(f"WGEN:OUTP ON")

    def gen_disable(self) -> None:
        """Disable waveform generator
        Returns:
            None
        """
        self.__write_to_dev(f"WGEN:OUTP OFF")

    def gen_function(self, waveform: str) -> bool:
        """Select type of waveform function to be generated (case-insensitive)
        Args:
            waveform: SINE/SIN for sine function;
                SQUARE/SQU for square function;
                RAMP for ramp function;
                DC for DC function;
                PULSE/PULS for pulse function;
                CARDINAL/SINC for cardinal sine function;
                ARBITRARY/ARB for arbitrary function
        Returns:
            True if waveform function is invalid
        """
        functions = {
            "SINE": "SIN", "SQUARE": "SQU", "RAMP": "RAMP", "DC": "DC", "PULSE": "PULS",
            "CARDINAL": "SINC", "ARBITRARY": "ARB"
        }
        for value in list(functions.values()):
            functions[value] = value    # make it possible to use the abbreviations as well
        if waveform.upper() not in functions:
            return True
        self.__write_to_dev(f"WGEN:FUNC {functions[waveform.upper()]}")
        return False

    def gen_frequency(self, frequency: float) -> None:
        """Set frequency of waveform
        Args:
            frequency: frequency in Hz, range dependent on function
        Returns:
            None
        """
        self.__write_to_dev(f"WGEN:FREQ {frequency:.3f}")
        

if __name__ == "__main__":
    d = DriverRTM3004()
    d.serial_start()
    d.get_id()
    d.do_reset()
    d.gen_enable()
    d.gen_function("RAMP")
    sleep(2)
    d.gen_function("SINE")
    sleep(1)
    d.gen_frequency(100)
    print(d.view_cmd_stack())
    d.serial_close()
    