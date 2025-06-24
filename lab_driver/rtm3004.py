from requests import options
from logging import getLogger
from lab_driver.mxo4x import *


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
            text_out = self.SerialDevice.query(order).strip("\0").strip()
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

    def test(self, cmd: str):
        """Test any command with the device
        Args:
            cmd: Some command to be sent to the device
        Returns:
            Result of the command if it was a query, None if it was a write command
        """
        if '?' in cmd:
            return self.__read_from_dev(cmd)
        else:
            self.__write_to_dev(cmd)

    def get_id(self, do_print=True) -> str:
        """Getting the device ID
        Args:
            do_print: optionally print the device ID to stdout
        Returns:
            Device ID as a string
        """
        # For some reason the ID ends on three null characters, so strip the string 
        id = self.__read_from_dev("*IDN?")
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

        list_dev = scan_instruments()
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

    def do_reset(self) -> None:
        """Reset the device, then wait two seconds
        Returns:
            None
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev("*RST")
            self.sync()
            sleep(1)
    
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

    def dig_technology(self, tech: str, logic_channel: int) -> bool:
        """Select threshold voltage for various types of circuits and apply it to the whole
        channel group the indicated logic channel belongs to
        Args:
            tech: "TTL" (1.4 V), "ECL" (-1.3 V), "CMOS" (2.5 V) or "MAN"/"MANUAL"
            logic_channel: 0..15
        Returns:
            True if tech is or logic channel invalid   
        """
        valid_techs = ("TTL", "ECL", "CMOS", "MAN", "MANUAL")
        if tech not in valid_techs or logic_channel not in range(16):
            return True
        self.__write_to_dev(f"DIG{logic_channel}:TECH {tech}")
        return False
    
    def dig_threshold(self, threshold: float, logic_channel: int) -> bool:
        """Set logical threshold for the nibble (D0...D3, D4...D7, D8...D11, and D12...D15)
        to which the logic channel belongs
        Args:
            threshold: Threshold level in volts
            logic_channel: 0..15
        Returns:
            True if logic channel is invalid
        """
        if logic_channel not in range(16):
            return True
        self.__write_to_dev(f"DIG{logic_channel}:THR {threshold}")
        return False
    
    def dig_enable(self, pod: int):
        """Enable a logic pod
        Args:
            pod: 1 or 2 for the corresponding logic pod
        Returns:
            True if pod number is invalid
        """
        if pod not in (1,2):
            return True
        self.__write_to_dev(f"LOG{pod}:STAT 1")
        return False
    
    def dig_disable(self, pod: int):
        """Disable a logic pod
        Args:
            pod: 1 or 2 for the corresponding logic pod
        Returns:
            True if pod number is invalid
        """
        if pod not in (1,2):
            return True
        self.__write_to_dev(f"LOG{pod}:STAT 0")
        return False
    
    def dig_hysteresis(self, level, logic_channel: int) -> bool:
        """Defines the level of the hysteresis to avoid the change of signal states due to noise.
        The setting applies to the logic pod to which the indicated logic channel belongs.
        Args:
            level: "SMALL", "MEDIUM", "LARGE" (case-insensitive) or 0, 1, 2 respectively.
            logic_channel: 0..15
        Returns:
            True if hysteresis level or logic channel is invalid
        """
        if type(level) is str and level.upper() not in ("SMALL", "MEDIUM", "LARGE"):
            return True
        if type(level) is int and (level not in range(3) or logic_channel not in range(16)):
            return True
        if type(level) == int:
            level = ("SMALL", "MEDIUM", "LARGE")[level]
        self.__write_to_dev(f"DIG{logic_channel}:HYST {level}")
        return False
    
    def trig_event_mode(self, sequence: bool) -> None:
        """Select whether to trigger on a single event or a sequence of A and B events.
        Args:
            sequence: True to enable sequence trigger of A and B, False for only an A trigger
        """
        self.__write_to_dev(f"TRIG:B:ENAB {sequence}")
        return False
    
    def trig_a_mode(self, mode: str):
        """Set the trigger mode, which determines device behaviour if no trigger occurs.
        Args:
            mode: "AUTO" or "NORM"/"NORMAL" (case-insensitive)
        Returns:
            True if trigger mode is invalid
        """
        if mode := mode.upper() not in ("AUTO", "NORM", "NORMAL"):
            return True
        self.__write_to_dev(f"TRIG:A:MODE {mode}")
        return False
    
    def trig_b_mode(self, mode: str) -> bool:
        """Set either a time or an event delay after an A trigger before recognising a B trigger
        Args:
            mode: "DELAY" or "EVENT" (case-insensitive)
        Returns:
            True if trigger mode is invalid
        """
        if mode := mode.upper() not in ("DELAY", "EVENT"):
            return True
        self.__write_to_dev(f"TRIG:B:MODE {mode}")
        return False
    
    def trig_source(self, source: str, event: int = 1) -> bool:
        """Set the trigger source of either the A or B trigger
        Args:
            source: "CH1" to "CH4", "D0" to "D15" for both triggers.
                A-trigger additionally allows "SBUS1", "SBUS2", "EXT" and "LINE".
            event: 1 = A-trigger, 2 = B-trigger
        Returns:
            True if trigger source is invalid for the given event or event is invalid
        """
        sources = [f"CH{i}" for i in range(1,5)] + [f"D{i}" for i in range(16)] + ["SBUS1", "SBUS2", "EXT", "LINE"]
        if event not in (1,2) or (event == 1 and source not in sources) or (event == 2 and source not in sources[:-4]):
            return True
        self.__write_to_dev(f"TRIG:{'A' if event == 1 else 'B'}:SOUR {source}")
        return False
    
    def trig_delay(self, delay: float) -> None:
        """Sets the time that the instrument waits after an A-trigger until it recognises B-triggers
        Args:
            delay: delay in seconds in range [20e-9, 6.871946854]
        Returns:
            None
        """
        delay = self.__clamp(20e-9, delay, 6.871946854)
        self.__write_to_dev(f"TRIG:B:DEL {delay}")

    def trig_b_trigger_count(self, count: int) -> None:
        """Number of B-trigger conditions that need to happen before the B-trigger is actually triggered.
        Args:
            count: number of times B-trigger must occur in sequence from 1 to 65535
        Returns:
            None
        """
        count = self.__clamp(1, count, 65535)
        self.__write_to_dev(f"TRIG:B:EVEN:COUNT {count}")

    def trig_find_level(self) -> None:
        """Automatically sets trigger level to half of signal amplitude
        Returns:
            None
        """
        self.__write_to_dev("TRIG:A:FIND")
    
    def trig_edge_lowpass(self, large: bool, small: bool) -> None:
        """Set an additional lowpass filter in the trigger path
        Args:
            large: True for a 100 MHz lowpass filter, False to disable
            small: True for a 5 KHz lowpass filter, False to disable
        Returns:
            None
        """
        self.__write_to_dev(f"TRIG:A:EDGE:FILT:NREJ {int(large)}")
        self.__write_to_dev(f"TRIG:A:EDGE:FILT:HFR {int(small)}")
    
    def trig_edge_noisereject(self, level: str | int) -> bool:
        """Sets a hysteresis range around the trigger level to avoid unwanted triggers by noise oscillations.
        The value of each hysteresis level depends on the vertical scale.
        Args:
            level: "AUTO" = 0,
                "SMALL" ("S") = 1,
                "MEDIUM" ("M") = 2,
                "LARGE" ("L") = 3
        Returns:
            True if level is invalid
        """
        options = ("AUTO", "SMALL", "MEDIUM", "LARGE", "S", "M", "L", 0, 1, 2, 3)
        if type(level) is str:
            level = level.upper()
            if len(level) == 1:
                level = options[options.index(level) - 3]
        if level not in options:
            return True
        if type(level) is int:
            level = options[level]
        self.__write_to_dev(f"TRIG:A:HYST {level}")
        return False

    def trig_edge_coupling(self, coupling: str) -> bool:
        """Sets the coupling for the trigger source (case-insensitive).
        Args:
            coupling:
                "DC" - Direct current coupling. The trigger signal remains unchanged.
                "AC" - Alternating current coupling. A highpass filter removes the
                DC offset voltage from the trigger signal.
                "LFR" - Sets the trigger coupling to high frequency. A 15 kHz highpass filter
                removes lower frequencies from the trigger signal. Use this mode only with
                very high frequency signals.
        Returns:
            True if coupling mode is invalid
        """
        if coupling := coupling.upper() not in ("AC", "DC", "LFR"):
            return True
        self.__write_to_dev(f"TRIG:A:EDGE:COUP {coupling}")
        return False

    def trig_edge_direction(self, direction: Threeway, event: int = 1) -> bool:
        """Set edge direction for trigger
        Args:
            direction: NEGATIVE for falling edge, POSITIVE for rising edge, NEUTRAL for either
            event: 1 = A-trigger, 2 = B-trigger
        Returns:
            True if direction or event is invalid
        """
        if direction not in (-1,0,1) or event not in (1,2):
            return True
        args = ["NEG", "EITH", "POS"]
        self.__write_to_dev(f"TRIG:{'A' if event == 1 else 'B'}:EDGE:SLOP {args[direction + 1]}")
        return False
    
    def trig_level(self, level: float, channel: int = 1) -> bool:
        """Set the trigger threshold voltage for edge, width, and timeout trigger
        Args:
            level: Depends on vertical scale, unit [V]
            channel: 1..4 are the corresponding analogue channels, 5 is external trigger input
        Returns:
            True if channel is invalid
        """
        if channel not in (1,2,3,4,5):
            return True
        self.__write_to_dev(f"TRIG:A:LEV{channel} {level}")
        return False

    def trig_export_source(self, source: str) -> bool:
        sources = ([f"CH{i}" for i in range(1, 5)]
            + ["D70", "D158"]
            + [f"MA{i}" for i in range(1, 6)]
            + [f"RE{i}" for i in range(1, 5)])
        if source not in sources:
            return True
        self.__write_to_dev(f"EXP:WAV:SOUR {source}")
        return False

    def trig_export_on_trigger(self, state: bool) -> None:
        """Decide whether the waveform is saved to a file on a trigger event
        Args:
            state: True to export waveform data on trigger, False to do nothing
        Returns:
            None
        """
        self.__write_to_dev(f"TRIG:EVEN {int(state)}")
        self.__write_to_dev(f"TRIG:EVEN:WFMS {int(state)}")
        
    def fra_enter(self):
        """Enter frequency response analysis mode. This is done automatically whenever an FRA function is called.
        Returns:
            None
        """
        self.__write_to_dev("SPEC ON")
        self.sync()
    
    def fra_exit(self):
        """Exit frequency response analysis mode
        Returns:
            None
        """
        self.__write_to_dev("SPEC OFF")
        self.sync()
    
    def fra_freq_start(self, freq: float) -> bool:
        """Set the start frequency of the sweep.
        NOTICE: This function is broken and should not be relied upon.
        Args:
            freq: Frequency in Hz
        Returns:
            None
        """
        self.fra_enter()
        self.__write_to_dev(f"SPEC:FREQ:STAR {freq}")
        return False
    
    def fra_freq_stop(self, freq: float) -> bool:
        """Set the stop frequency of the sweep
        NOTICE: This function is broken and should not be relied upon.
        Args:
            freq: Frequency in Hz
        Returns:
            None
        """
        self.fra_enter()
        self.__write_to_dev(f"SPEC:FREQ:STOP {freq}")
        return False
    
    def fra_input_channel(self, channel: int) -> bool:
        """Set the channel used for the input signal of the device
        Args:
            channel: 1 to 4
        Returns:
            True for invalid channel number
        """
        if channel not in (1,2,3,4):
            return True
        self.fra_enter()
        self.__write_to_dev(f"SPEC:SOUR CH{channel}")
        return False
