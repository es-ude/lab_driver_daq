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

    def live_command_mode(self) -> None:
        """DEBUGGING - enter statements during the execution of the program using the Python
        interpreter. Results and errors are printed.
        Returns:
            None
        """
        print(">> LIVE COMMAND MODE")
        print(">> Type 'exit' to stop.")
        while (cmd := input("> ")).strip() != "exit":
            try:
                output = {}
                exec(f"output = {cmd}", globals(), output)
                if output["output"] is not None:
                    print(output["output"])
            except Exception as e:
                print(e)
                print(">> Command failed. Try again.")
        print(">> END OF LIVE COMMAND MODE")

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
    
    def scale_vertical(self, scale: float) -> bool:
        """Sets the vertical scale (V/div) of all channels on the GUI
        Args:
            scale: [0.001,10] Volts per division
        Returns:
            True if scale is out of range
        """
        if not (0.001 <= scale <= 10):
            return True
        self.__write_to_dev(f"CHAN:SCAL {scale}")
        return False
    
    def scale_horizontal(self, scale: float) -> bool:
        """Sets the horizontal (time) scale of all channels on the GUI
        Args:
            scale: [1e-9,50] seconds, 1 ns precision
        Returns:
            True if scale is out of range
        """
        if not (1e-9 <= scale <= 50):
            return True
        self.__write_to_dev(f"TIM:SCAL {scale:.9f}")
        return False

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

    def gen_amplitude(self, amplitude: float) -> bool:
        """Set amplitude of waveform
        Args:
            amplitude: amplitude in volt from [0.06,6], 0.01 increment
        Returns:
            True if amplitude out of range
        """
        if not (0.06 <= amplitude <= 6):
            return True
        self.__write_to_dev(f"WGEN:VOLT {amplitude:.2f}")
        return False

    def gen_offset(self, offset: float) -> bool:
        """Set vertical offset of generated waveform
        Args:
            offset: vertical offset in volt from [-5,+5], 0.0001 increment
        Returns:
            True if offset out of range
        """
        if not (-5 <= offset <= 5):
            return True
        self.__write_to_dev(f"WGEN:VOLT:OFFS {offset:.4f}")
        return False    

    def gen_preset(self) -> None:
        """Preset the generator to a default setup including following settings:
        Sine wavefunction, 1 MHz frequency, 1 Vpp amplitude, 500 ns horizontal scale, 0.5 V/div vertical scale  
        Returns:
            None
        """
        self.gen_function("SINE")
        self.gen_frequency(1*MHz)
        self.gen_amplitude(1)
        self.scale_horizontal(5e-7)
        self.scale_vertical(.5)
    
    def sweep_freq_start(self, freq: float) -> None:
        self.__write_to_dev(f"WGEN:SWE:FST {freq}")
    
    def sweep_freq_stop(self, freq: float) -> None:
        self.__write_to_dev(f"WGEN:SWE:FEND {freq}")
    
    def sweep_duration(self, duration: float) -> None:
        self.__write_to_dev(f"WGEN:SWE:TIME {duration:.3f}")
    
    def sweep_type(self, shape: str) -> bool:
        if shape := shape.upper() not in ("LINEAR", "LOGARITHMIC", "TRIANGLE", "LIN", "LOG", "TRI"):
            return True
        self.__write_to_dev(f"WGEN:SWE:TYPE {shape}")
        return False
    
    def sweep_run(self):
        self.__write_to_dev(f"WGEN:SWE ON")
    
    def sweep_stop(self):
        self.__write_to_dev(f"WGEN:SWE OFF")
        

if __name__ == "__main__":
    d = DriverRTM3004()
    d.serial_start()
    d.get_id()
    d.do_reset()
    d.gen_enable()
    d.gen_preset()
    d.live_command_mode()
    d.serial_close()
    