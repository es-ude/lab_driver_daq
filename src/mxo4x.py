from time import sleep
import pyvisa
import platform
import sys
from RsInstrument import RsInstrument

mHz = .001
KHz = 1000
MHz = 1000000

# starting with Python 3.12, we may use these typedef-esque statements
if sys.version_info[:2] >= (3,12):
    # Threeway type is like a boolean, but with 3 states -1,0,1
    type Threeway = int
else:
    Threeway = int
LEFT: Threeway = -1
MIDDLE: Threeway = 0
RIGHT: Threeway = 1
NEGATIVE: Threeway = -1
NEUTRAL: Threeway = 0
POSITIVE: Threeway = 1
LOW: Threeway = -1
ZERO: Threeway = 0
HIGH: Threeway = 1
OFF: Threeway = 0



def scan_instruments(do_print=True) -> list:
    """Scanning the VISA bus for instruments
    Args:
        do_print: True to print every detected instrument
    Returns:
        List of all detected instruments
    """
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


class DriverMXO4X:
    """Class for handling the Rohde and Schwarz Mixed-Signal Oscilloscope MXO44 in Python"""
    SerialDevice: pyvisa.Resource | RsInstrument
    SerialActive = False
    _device_name_chck = "MXO"
    _gen_index = 1      # which generator to configure if none is explicitly stated
    _logic_group = 1    # which logic group to configure if none is explicitly stated
    _output_config = 1  # which screenshot output configuration to use if none is explicitly stated
    _trig_seq = False   # is trigger source sequence (else single)?
    _cmd_stack = []     # all executed commands are stored here LIFO for debugging purposes
    _firmware_version: str
    src_analogue = tuple(f"C{i}" for i in range(1,5))
    src_digital = tuple(f"D{i}" for i in range(16))
    src_math = tuple(f"M{i}" for i in range(1,6))
    src_reference = tuple(f"R{i}" for i in range(1,5))
    src_specmax = tuple(f"SPECMAXH{i}" for i in range(1,5))
    src_specmin = tuple(f"SPECMINH{i}" for i in range(1,5))
    src_specnorm = tuple(f"SPECNORM{i}" for i in range(1,5))
    src_specaver = tuple(f"SPECAVER{i}" for i in range(1,5))
    src_spectrum = src_specmax + src_specmin + src_specnorm + src_specaver
    src_all = src_analogue + src_digital + src_math + src_reference + src_spectrum
    src_groups = (src_analogue, src_digital, src_math, src_reference,
                  src_specmax, src_specmin, src_specnorm, src_specaver)

    def __init__(self):
        pass

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
    
    def view_cmd_stack(self, entries: int = 0):
        i = 1
        for cmd in self._cmd_stack[-entries:][::-1]:
            if type(cmd) == str:
                print(f"{i:03}> {cmd}")
            else:
                cmd, out = cmd[0], cmd[1]
                print(f"{i:03}> {cmd}\n     >> {out}")
            i += 1

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
            self._trig_seq = self.__read_from_dev("TRIG:MEV:MODE?") == "SEQ"
            # This command doesn't seem to exist? Windows doesn't throw exceptions on wrong commands,
            # that's why it worked there, but not on Linux
            # self.__write_to_dev("SYST:MIX")   # Instrument error detected: -113,"Undefined header;SYST:MIX"
            print(f"Right device is selected with: {self.get_id(False)}")
        else:
            print("Not right selected device. Please check!")

    def __do_check_idn(self) -> None:
        """Checking the IDN"""
        id_back = self.get_id(False)
        self.SerialActive = self._device_name_chck in id_back
        if self.SerialActive:
            self._firmware_version = id_back.split(',')[-1]

    def __fix_gen_index(self, gen_index: int) -> int:
        if gen_index is None or gen_index not in (1,2):
            return self._gen_index
        else:
            return gen_index

    def __fix_logic_index(self, logic_group: int) -> int:
        if logic_group is None or logic_group not in (1,2,3,4):
            return self._logic_group
        else:
            return logic_group

    def __clamp(self, x, y, z):
        return min(max(x, y), z)
    
    def sync(self, timeout = 86400000) -> None:
        """Wait until all queued commands have been processed
        Args:
            timeout: timeout in milliseconds, VISA exception thrown on timeout, default 1 day
        Returns:
            None
        """
        backup_timeout = self.SerialDevice.visa_timeout
        self.SerialDevice.visa_timeout = timeout
        self.__write_to_dev("*WAI")
        self.SerialDevice.visa_timeout = backup_timeout

    def serial_open_known_target(self, resource_name: str, do_reset=False) -> None:
        """Open the serial connection to device
        Args:
            resource_name: name of the device
            do_reset: reset device during initialisation
        Returns:
            None
        """
        if platform.system() == "Linux":
            try:
                self.SerialDevice = RsInstrument(resource_name)
            except:
                print(f"Could not find or open device {resource_name}")
                return
        else:
            rm = pyvisa.ResourceManager()
            self.SerialDevice = rm.open_resource(resource_name)

        self.__do_check_idn()
        self.__init_dev(do_reset)

    def serial_start(self, do_reset=False) -> None:
        """Open the serial connection to device if it is found
        Args:
            do_reset: reset device during initialisation
        Returns:
            None
        """
        if platform.system() == "Linux":
            # Resource string for MXO44
            self.serial_open_known_target("USB0::0x0AAD::0x0197::1335.5050k04-201451::INSTR", do_reset)
            return

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
        self.__init_dev(do_reset)

    def serial_close(self) -> None:
        """Close the serial connection
        Returns:
            None
        """
        self.SerialDevice.close()
        self.SerialActive = False

    def get_id(self, do_print=True) -> str:
        """Getting the device ID
        Args:
            do_print: optionally print the device ID to stdout
        Returns:
            Device ID as a string
        """
        id = self.__read_from_dev("*IDN?")
        if do_print:
            print(id)
        return id

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

    def change_display_mode(self, show_display: bool) -> None:
        """Decide whether display is shown during remote control
        Args:
            show_display: True to show display, False to show static image (may improve performance)
        Returns:
            None
        """
        self.__write_to_dev(f"SYST:DISP:UPD {int(show_display)}")
        self.sync()

    def change_remote_text(self, text: str) -> None:
        """Display an additional text in remote control
        Args:
            text: text to display
        Returns:
            None
        """
        self.__write_to_dev(f"SYST:DISP:MESS:STAT ON")
        self.__write_to_dev(f"SYST:DISP:MESS '{text}'")

    def gen_set_default_index(self, gen_index: int) -> bool:
        """Set which generator is affected by any of the gen_* functions by default
        Args:
            gen_index: index of new default generator: 1 or 2
        Returns:
            True if generator index is not 1 or 2
        """
        if gen_index not in (1,2):
            return True
        self._gen_index = gen_index
        return False

    def gen_get_default_index(self) -> int:
        """Get which generator is currently set as the default
        Returns:
            Default generator's index: 1 or 2
        """
        return self._gen_index

    def gen_enable(self, gen_index: int = None) -> None:
        """Enable waveform generator
        Args:
            gen_index: index of the target generator to affect (None for default)
        Returns:
            None
        """
        gen_index = self.__fix_gen_index(gen_index)
        self.__write_to_dev(f"WGEN{gen_index} ON")

    def gen_disable(self, gen_index: int = None) -> None:
        """Disable waveform generator
        Args:
            gen_index: index of the target generator to affect (None for default)
        Returns:
            None
        """
        gen_index = self.__fix_gen_index(gen_index)
        self.__write_to_dev(f"WGEN{gen_index} OFF")

    def gen_function(self, waveform: str, gen_index: int = None) -> bool:
        """Select type of waveform function to be generated (case-insensitive)
        Args:
            waveform: SINE/SIN for sine function;
                SQUARE/SQU for square function;
                RAMP for ramp function;
                DC for DC function;
                PULSE/PULS for pulse function;
                CARDINAL/SINC for cardinal sine function;
                CARDIAC/CARD for cardiac function;
                GAUSS/GAUS for gaussian function;
                LORENTZ/LORN for lorentz function;
                EXP RISE/EXPR for exponential rise function;
                EXP FALL/EXPF for exponential fall function;
                ARBITRARY/ARB for arbitrary function
            gen_index: index of the target generator to affect (None for default)
        Returns:
            True if waveform function is invalid
        """
        functions = {
            "SINE": "SIN", "SQUARE": "SQU", "RAMP": "RAMP", "DC": "DC", "PULSE": "PULS",
            "CARDINAL": "SINC", "CARDIAC": "CARD", "GAUSS": "GAUS", "LORENTZ": "LORN",
            "EXP RISE": "EXPR", "EXP FALL": "EXPF", "ARBITRARY": "ARB"
        }
        for value in list(functions.values()):
            functions[value] = value    # make it possible to use the abbreviations as well
        if waveform.upper() not in functions:
            return True
        gen_index = self.__fix_gen_index(gen_index)
        self.__write_to_dev(f"WGEN{gen_index}:FUNC {functions[waveform.upper()]}")
        return False

    def gen_frequency(self, frequency: float, gen_index: int = None) -> None:
        """Set frequency of waveform
        Args:
            frequency: frequency in Hz from [0.001,10^8], 0.001 increment
            gen_index: index of the target generator to affect (None for default)
        Returns:
            None
        """
        gen_index = self.__fix_gen_index(gen_index)
        self.__write_to_dev(f"WGEN{gen_index}:FREQ {frequency:.3f}")

    def gen_amplitude(self, amplitude: float, gen_index: int = None) -> bool:
        """Set amplitude of waveform
        Args:
            amplitude: amplitude in volt from [0.01,12], 0.01 increment
            gen_index: index of the target generator to affect (None for default)
        Returns:
            True if amplitude out of range
        """
        if not (0.01 <= amplitude <= 12):
            return True
        amplitude /= 1.08   # constant factor to fix offset
        gen_index = self.__fix_gen_index(gen_index)
        self.__write_to_dev(f"WGEN{gen_index}:VOLT {amplitude:.2f}")
        return False

    def gen_offset(self, offset: float, gen_index: int = None) -> bool:
        """Set vertical offset of generated waveform
        Args:
            offset: vertical offset in volt from [-5.995,+5.995], 0.01 increment
            gen_index: index of the target generator to affect (None for default)
        Returns:
            True if offset out of range
        """
        if not (-5.995 <= offset <= 5.995):
            return True
        gen_index = self.__fix_gen_index(gen_index)
        self.__write_to_dev(f"WGEN{gen_index}:VOLT:OFFS {offset:.2f}")
        self.__write_to_dev(f"WGEN{gen_index}:VOLT:DCL {offset:.2f}")
        return False

    def gen_preset(self, gen_index: int = None) -> None:
        """Preset the generator to a default setup including following settings:
        Sine wavefunction, 1 MHz frequency and 1 Vpp amplitude
        Args:
            gen_index: index of the target generator to affect (None for default)
        Returns:
            None
        """
        gen_index = self.__fix_gen_index(gen_index)
        self.__write_to_dev(f"WGEN{gen_index}:PRES")

    def dig_set_default_logic_group(self, logic_group: int) -> bool:
        """Set the default logic group that shall be configured by dig_* functions
        Args:
            logic_group: default logic group (1-4)
        Returns:
            True if given group does not exist
        """
        if logic_group not in (1,2,3,4):
            return True
        self._logic_group = logic_group
        return False

    def dig_get_default_logic_group(self) -> int:
        """Get the currently set default logic group
        Returns:
            Logic group from 1 to 4
        """
        return self._logic_group

    def dig_technology(self, tech, logic_group: int = None) -> bool:
        """Select threshold voltage for various types of circuits and apply to all digital channels
        Args:
            tech:
                 15: TTL
                 25: CMOS 5.0 V
                165: CMOS 3.3 V
                125: CMOS 2.5 V
                  9: CMOS 1.85 V
                -13: ECL, -1.3 V
                 38: PECL
                 20: LVPECL
                  0: Ground
            logic_group: index of logic group to configure
        Returns:
            True if selected technology is unsupported
        """
        valid_techs = (15,25,165,125,9,-13,38,20,0)
        if tech not in valid_techs:
            return True
        logic_group = self.__fix_logic_index(logic_group)
        cmd = f"PBUS{logic_group}:TECH "
        if tech == 9:
            self.__write_to_dev(cmd + "V09")
        elif tech == -13:
            self.__write_to_dev(cmd + "VM13")
        else:
            self.__write_to_dev(cmd + f"V{tech}")
        return False

    def dig_threshold(self, threshold: float, channel_group: int = 1, logic_group: int = None) -> bool:
        """Manually set a logical threshold voltage for some channel group
        Args:
            threshold: threshold voltage in volt in range [-8,+8]
            channel_group: 1 = digital channels 0..3
                2 = digital channels 4..7
                3 = digital channels 8..11
                4 = digital channels 12..15
                No channel group needed when coupling is enabled
            logic_group: index of logic group to configure
        Returns:
            True if channel group is invalid or threshold out of range
        """
        if channel_group not in (1,2,3,4) or not (-8 <= threshold <= 8):
            return True
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:THR{channel_group} {threshold}")
        return False

    def __dig_activation_state(self, bits, enable: bool, logic_group: int = None) -> bool:
        logic_group = self.__fix_logic_index(logic_group)
        if type(bits) == int and bits in range(16):
            self.__write_to_dev(f"PBUS{logic_group}:BIT{bits} {int(enable)}")
            return False
        else:
            try:
                for bit in filter(lambda x: x in range(16), bits):
                    self.__write_to_dev(f"PBUS{logic_group}:BIT{bit} {int(enable)}")
                return any(x not in range(16) for x in bits)
            except:
                return True

    def dig_enable(self, bits, logic_group: int = None) -> bool:
        """Enable logic channels
        Args:
            bits: Either an int or an iterable of integers denoting the bits to enable
            logic_group: index of logic group to configure
        Returns:
            True if any bit indices are invalid (valid bits are applied)
        """
        return self.__dig_activation_state(bits, True, logic_group)

    def dig_disable(self, bits, logic_group: int = None) -> bool:
        """Disable logic channels
        Args:
            bits: Either an int or an iterable of integers denoting the bits to disable
            logic_group: index of logic group to configure
        Returns:
            True if any bit indices are invalid (valid bits are applied)
        """
        return self.__dig_activation_state(bits, False, logic_group)

    def dig_show_dig_signals(self, logic_group: int = None) -> None:
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:DISP:SHDI ON")

    def dig_hide_dig_signals(self, logic_group: int = None) -> None:
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:DISP:SHDI OFF")

    def dig_hysteresis(self, level, channel_group: int = 1, logic_group: int = None) -> bool:
        """Set hysteresis size for channels
        Args:
            level: level of hysteresis "NORMAL", "ROBUST", "MAXIMUM" or "SMALL",
                "MEDIUM", "LARGE" or 0, 1, 2; strings are case-insensitive
            channel_group: 1 = digital channels 0..3
                2 = digital channels 4..7
                3 = digital channels 8..11
                4 = digital channels 12..15
                No channel group needed when coupling is enabled
        Returns:
            True if channel group or hysteresis is invalid
        """
        if channel_group not in (1,2,3,4):
            return True
        logic_group = self.__fix_logic_index(logic_group)
        levels_str = ["NORMAL", "ROBUST", "MAXIMUM"]
        levels_alt_str = ["SMALL", "MEDIUM", "LARGE"]
        if type(level) == int and level in range(3):
            self.__write_to_dev(f"PBUS{logic_group}:HYST{channel_group} {levels_str[level]}")
        elif type(level) == str and level.upper() in levels_str:
            self.__write_to_dev(f"PBUS{logic_group}:HYST{channel_group} {level.upper()}")
        elif type(level) == str and level.upper() in levels_alt_str:
            normalised_level = levels_str[levels_alt_str.index(level.upper())]
            self.__write_to_dev(f"PBUS{logic_group}:HYST{channel_group} {normalised_level}")
        else:
            return True
        return False

    def dig_hysteresis_coupling(self, state: bool, logic_group: int = None) -> None:
        """Set the threshold and hysteresis for all digital channels and all buses to the same value
        Args:
            state: True to couple all levels
        Returns:
            None
        """
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:THC {int(state)}")

    def dig_show_bus(self, logic_group: int = None) -> None:
        """Show the bus signal and values in the diagram
        Returns:
            None
        """
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:DISP:SHBU ON")

    def dig_hide_bus(self, logic_group: int = None) -> None:
        """Hide the bus signal and values in the diagram
        Returns:
            None
        """
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:DISP:SHBU OFF")

    def dig_bus_data_format(self, format: str, logic_group: int = None) -> bool:
        """Set the data format for bus values, which are displayed in the decode table and the comb
        bus display
        Args:
            format: "HEX", "OCT", "BIN", "ASCII", "SIGNED" or "UNSIGNED" (case-insensitive)
        Returns:
            True if data format is invalid
        """
        if (format := format.upper()) not in ("HEX", "OCT", "BIN", "ASCII", "SIGNED", "UNSIGNED"):
            return True
        if format == "SIGNED":
            format = "SIGN"
        elif format == "UNSIGNED":
            format = "USIG"
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:DATA:FORM {format}")
        return False

    def dig_bus_header(self, logic_group: int = None) -> str:
        """Get the header data of the indicated bus
        Returns:
            In order - XStart, acquisition time before trigger, in s;
                XStop, acquisition time after trigger, in s;
                Record length of the waveform in Samples;
                Number of values per sample interval, which is 1 for digital data.
        """
        logic_group = self.__fix_logic_index(logic_group)
        return self.__read_from_dev(f"PBUS{logic_group}:DATA:HEAD?")

    def __fix_output_config(self, output_config: int) -> int:
        return output_config if output_config in (1,2) else self._output_config

    def sshot_get_default_config(self):
        """Get the number of the default screenshot output configuration
        Returns:
            Output configuration number
        """
        return self._output_config

    def sshot_set_default_config(self, output_config: int) -> bool:
        """Set which screenshot output configuration is used per default
        Args:
            output_config: 1 or 2
        Returns:
            True if output config is invalid
        """
        if output_config in (1, 2):
            self._output_config = output_config
            return False
        return True

    def sshot_get_filename(self) -> str:
        """A string of the path and filename of screenshots
        Returns:
            Path string
        """
        return self.__read_from_dev("MMEM:NAME?")

    def sshot_set_filename(self, filename: str) -> None:
        """Set filename and path of screenshots
        Args:
            filename: Path string
        Returns:
            None
        """
        self.__write_to_dev(f"MMEM:NAME {filename}")

    def sshot_destination(self, dest: str, output_config: int = None) -> bool:
        """Select whether to save screenshot in a file or clipboard
        Args:
            dest: "FILE" or "CLIPBOARD" (case-insensitive)
        Returns:
            True if destination is invalid
        """
        output_config = self.__fix_output_config(output_config)
        if dest.upper() == "FILE":
            self.__write_to_dev(f"HCOP:DEST{output_config} MMEM")
        elif dest.upper() == "CLIPBOARD":
            self.__write_to_dev(f"HCOP:DEST{output_config} CLIPBOARD")
        else:
            return True
        return False

    def sshot_file_format(self, format: str, output_config: int = None) -> bool:
        """Screenshot file format when saving to a file
        Args:
            format: "JPG" or "PNG" (case-insensitive)
        Returns:
            True if format is invalid
        """
        output_config = self.__fix_output_config(output_config)
        if format.upper() not in ("PNG", "JPG"):
            return True
        self.__write_to_dev(f"HCOP:DEV{output_config}:LANG {format.upper()}")
        return False

    def sshot_invert_colours(self, state: bool, output_config: int = None) -> None:
        """Invert all colours of the screenshot
        Args:
            state: True to invert colours, False to leave it unchanged
        Returns:
            None
        """
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:DEV{output_config}:INV {int(state)}")

    def sshot_white_background(self, state: bool, output_config: int = None) -> None:
        """Invert only the background colour so it appears white in a screenshot
        Args:
            state: True for white background, False for black
        Returns:
            None
        """
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:DEV{output_config}:WBKG {int(state)}")

    def sshot_include_signal_bar(self, state: bool, output_config: int = None) -> None:
        """Include the signal bar below the diagram area in a screenshot
        Args:
            state: True to include signal bar, False to hide it
        Returns:
            None
        """
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:DEV{output_config}:ISBA {int(state)}")

    def sshot_include_dialog_box(self, state: bool, output_config: int = None) -> None:
        """Include any open dialog box in a screenshot
        Args:
            state: True to include dialog boxes on screenshots, False to hide them 
        Returns:
            None
        """
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:DEV{output_config}:SSD {int(state)}")

    def sshot_capture_now(self, output_config: int = None) -> None:
        """Start immediate output of the display image to a screenshot, the display is automatically
        enabled if it's currently showing a static image
        Returns:
            None
        """
        if int(self.__read_from_dev("SYST:DISP:UPD?")) == 0:
            self.change_display_mode(True)
            self.sync()
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:IMM{output_config}")

    def sshot_capture_next(self, output_config: int = None) -> None:
        """Start output of the next display image to a screenshot, the display is automatically
        enabled if it's currently showing a static image
        Returns:
            None
        """
        if int(self.__read_from_dev("SYST:DISP:UPD?")) == 0:
            self.change_display_mode(True)
            self.sync()
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:IMM{output_config}:NEXT")

    def trig_event_mode(self, mode: str) -> bool:
        """Select whether to trigger on a single event or a sequence of events
        Args:
            mode: "SINGLE" or "SEQUENCE" (case-insensitive)
        Returns:
            True when mode is invalid
        """
        if (mode := mode.upper()) not in ("SINGLE", "SEQUENCE"):
            return True
        self.__write_to_dev(f"TRIG:MEV:MODE {mode}")
        return False

    def trig_source(self, source: str, event: int = 1) -> bool:
        """Select the source of the trigger signal. Sequence trigger mode only allows analog sources.
        Args:
            source: C1, C2, C3, C4 for single or sequence event mode
                EXT, LINE, Dx for x in [0..15], SBUS1, SBUS2, SBUS3, SBUS4 only for single event mode
            event: 1 = A-trigger, 2 = B-trigger, 3 = reset event (for sequence trigger)
        Returns:
            True if source or event invalid
        """
        sources = ([f"C{i}" for i in range(1, 5)] + [f"D{i}" for i in range(16)]
                   + [f"SBUS{i}" for i in range(1, 5)] + ["EXT", "LINE"])
        if (self._trig_seq and source not in sources[:4]) or (source not in sources) or (event not in (1,2,3)):
            return True
        self.__write_to_dev(f"TRIG:EVEN{event}:SOUR {source}")
        return False

    def trig_delay(self, delay: float) -> None:
        """Sets the time that the instrument waits after an A-trigger until it recognises B-triggers
        Args:
            delay: delay in seconds
        Returns:
            None
        """
        delay = self.__clamp(0, delay, 50)
        self.__write_to_dev(f"TRIG:MEV:SEQ1:DEL {delay}")

    def trig_b_trigger_count(self, count: int) -> None:
        """
        Args:
            count: number of times B-trigger must occur in sequence from 1 to 2147483647
        Returns:
            None
        """
        count = self.__clamp(1, count, (1 << 31) - 1)
        self.__write_to_dev(f"TRIG:MEV:SEQ1:COUN {count}")

    def trig_toggle_reset_event(self, state: bool) -> None:
        """Enable or disable the reset event in sequence event mode
        Args:
            state: True to enable, False to disable
        Returns:
            None
        """
        self.__write_to_dev(f"TRIG:MEV:SEQ3:RES:EVEN {int(state)}")

    def trig_toggle_reset_event_timeout(self, state: bool) -> None:
        """Toggle whether event sequence shall time out when not receiving enough B-triggers in time
        Args:
            state: True to enable reset event by timeout, False to disable
        Returns:
            None
        """
        self.__write_to_dev(f"TRIG:MEV:SEQ:RES:TIM {int(state)}")

    def trig_reset_event_timeout_time(self, timeout: float) -> None:
        """Set the time to elapse before reset event by timeout is triggered
        Args:
            timeout: Time in seconds from 0 to 50
        Returns:
            None
        """
        timeout = self.__clamp(0, timeout, 50)
        self.__write_to_dev(f"TRIG:MEV:SEQ:RES:TIM:TIME {timeout}")

    def trig_sequence_type(self, type: str) -> bool:
        """Select the type of the trigger sequence
        Args:
            type: "A" for single event mode,
                "ABR" for sequence A → B → R,
                "AZ" for sequence A → Zone trigger,
                "ASB" for sequence A → Serial bus
        Returns:
            True if sequence type is invalid
        """
        if type not in ("A", "ABR", "AZ", "ASB"):
            return True
        if type == "A":
            type = "AONL"
        self.__write_to_dev("TRIG:MEV:AEV " + type)
        return False
    
    def trig_level(self, level: float, channel: int = 1, event: int = 1) -> bool:
        """Sets the trigger level for the specified event and source (channel).
        If the trigger source is serial bus, the trigger level is set by the
        thresholds in the protocol configuration.
        Args:
            level: -10 to 10 volts, value is clamped
            channel: 1 to 4, index of analogue channel
            event: 1 = A-trigger, 2 = B-trigger, 3 = reset event (for sequence trigger)
        Returns:
            True if channel or event is invalid
        """
        if channel not in (1,2,3,4) or event not in (1,2,3):
            return True
        level = self.__clamp(-10, level, 10)
        self.__write_to_dev(f"TRIG:EVEN{event}:LEV{channel} {level:.3}")
        return False
    
    def trig_find_level(self) -> None:
        """Automatically sets trigger level to 0.5 * (MaxPeak - MinPeak).
        In a trigger sequence, all events (A, B and R) are affected.
        This function does not work for trigger sources Extern and Line.
        Returns:
            None
        """
        self.__write_to_dev("TRIG:FIND")
    
    def trig_edge_direction(self, direction: Threeway, event: int = 1) -> bool:
        """Set edge direction for trigger
        Args:
            direction: NEGATIVE for falling edge, POSITIVE for rising edge,
                NEUTRAL for either.
            event: 1 = A-trigger, 2 = B-trigger, 3 = reset event (for sequence trigger)
        Returns:
            True if direction or event is invalid
        """
        if direction not in (-1,0,1) or event not in (1,2,3):
            return True
        args = ["NEG", "EITH", "POS"]
        self.__write_to_dev(f"TRIG:EVEN{event}:EDGE:SLOP {args[direction + 1]}")
        return False
    
    def trig_edge_level(self, level: float) -> None:
        """Set external trigger source trigger level
        Args:
            level: -5 to 5 volts, value is clamped
        Returns:
            None 
        """
        level = self.__clamp(-5, level, 5)
        self.__write_to_dev(f"TRIG:ANED:LEV {level}")
    
    def trig_edge_coupling(self, coupling: str) -> bool:
        """Sets the connection of the external trigger signal, i.e. the
        input impedance and a termination. The coupling determines what
        part of the signal is used for triggering.
        Args:
            coupling:
                "DC" - Connection with 50 Ω termination, passes both DC
                and AC components of the signal.
                "DCLimit" - Connection with 1 MΩ termination, passes both
                DC and AC components of the signal.
                "AC" - Connection with 1 MΩ termination through DC capacitor,
                removes DC and very low-frequency components. The waveform
                is centered on zero volts. 
        Returns:
            True if coupling mode is invalid
        """
        if coupling not in ("AC", "DC", "DCLimit"):
            return True
        self.__write_to_dev(f"TRIG:ANED:COUP {coupling}")
        return False
    
    def trig_edge_filter(self, filter: Threeway) -> bool:
        """Select filter mode for external signal
        Args:
            filter: LOW for lowpass filter, HIGH for highpass filter, OFF to disable filter
        Returns:
            True if filter mode is invalid
        """
        if filter not in (-1,0,1):
            return True
        args = ["RFR", "OFF", "LFR"]
        self.__write_to_dev(f"TRIG:ANED:FILT {args[filter + 1]}")
        return False
    
    def trig_edge_highpass(self, cutoff: int) -> bool:
        """Frequencies below the cutoff frequency are rejected,
        higher frequencies pass the filter
        Args:
            cutoff: 5 or 50 (unit: KHz)
        Returns:
            True if cutoff frequency is invalid
        """
        if cutoff not in (5,50):
            return True
        self.__write_to_dev(f"TRIG:ANED:CUT:HIGH KHZ{cutoff}")
        return False

    def trig_edge_lowpass(self, cutoff: int) -> bool:
        """Frequencies higher than the cutoff frequency are rejected,
        lower frequencies pass the filter
        Args:
            cutoff: 50 or 50000 (unit: KHz)
        Returns:
            True if cutoff frequency is invalid
        """
        if cutoff not in (50, 50000):
            return True
        unit = 'K' if cutoff == 50 else 'M'
        self.__write_to_dev(f"TRIG:ANED:CUT:LOWP {unit}HZ50")
        return False
    
    def trig_edge_noisereject(self, state: bool) -> None:
        """Enable an automatic hysteresis on the trigger level to
        avoid unwanted trigger events caused by noise.
        Args:
            state: True for noise rejection, False to disable
        Returns:
            None
        """
        self.__write_to_dev(f"TRIG:ANED:NREJ {int(state)}")
    
    def trig_export_on_trigger(self, state: bool) -> None:
        """Decide whether the waveform is saved to a file on a trigger event
        Args:
            state: True to export waveform data on trigger, False to do nothing
        Returns:
            None
        """
        self.__write_to_dev(f"TRIG:ACT:WFMS {"TRIG" if state else "NOAC"}")
    
    def is_source_active(self, source: str) -> bool:
        if source not in self.src_all:
            return False
        if self._firmware_version < "2.2.2.1" and source in self.src_spectrum and source[-1] > '1':
            return False    # support for 4 spectrums only since 2.2.2.1
        # determine which group the source belongs to, there can only be one, so index 0
        src_group = tuple(list(filter(lambda xs: source in xs, self.src_groups))[0])
        # get the correct command structure according to the source group, then
        # insert the channel number taken from the source string into the {} placeholder
        return bool(int(self.__read_from_dev({
            self.src_analogue: "CHAN{}:STAT?",
            self.src_digital: "DIG{}:STAT?",
            self.src_math: "CALC:MATH{}:STAT?",
            self.src_reference: "REFC{}:STAT?",
            self.src_specnorm: "CALC:SPEC{}:STAT?",
            self.src_specaver: "CALC:SPEC{}:STAT?",
            self.src_specmin: "CALC:SPEC{}:STAT?",
            self.src_specmax: "CALC:SPEC{}:STAT?",
        }[src_group].format("".join(filter(str.isnumeric, source))))))
        
    
    def export_scope(self, scope: str) -> bool:
        """Defines the part of the waveform record that will be stored
        Args:
            scope: (case-insensitive)
                "DISPLAY" - waveform data that is displayed in the diagram.
                "ALL" - entire waveform, usually larger than what is displayed.
                "CURSOR" - data between the cursor lines if a cursor measurement
                    is defined for the source waveform.
                "GATE" - data included in the measurement gate if a gated
                    measurement is defined for the source waveform.
                "MANUAL" - data between user-defined start and stop values.
        Returns:
            True if scope is invalid
        """
        if (scope := scope.upper()) not in ["DISPLAY", "ALL", "CURSOR", "GATE", "MANUAL"]:
            return True
        self.__write_to_dev(f"EXP:WAV:SCOP {scope}")
        return False
    
    def export_manual_start(self, time) -> None:
        """Set the start time value for waveform export in MANUAL mode
        Args:
            time: start time from -1e26 to +1e26 in seconds with 2 decimal precision
                (value is clamped to fit range)
        Returns:
            None
        """
        time = self.__clamp(-1e26, time, 1e26)
        self.__write_to_dev(f"EXP:WAV:STAR {time:.2}")
    
    def export_manual_stop(self, time) -> None:
        """Set the end time value for waveform export in MANUAL mode
        Args:
            time: end time from -1e26 to +1e26 in seconds with 2 decimal precision
                (value is clamped to fit range)
        Returns:
            None
        """
        time = self.__clamp(-1e26, time, 1e26)
        self.__write_to_dev(f"EXP:WAV:STOP {time:.2}")
    
    def export_save(self) -> None:
        """Save the waveform to the specified file
        Returns:
            None
        """
        self.__write_to_dev("EXP:WAV:SAVE")
    
    def export_abort(self) -> None:
        """Abort a running export started by export_save()
        Returns:
            None
        """
        self.__write_to_dev("EXP:WAV:ABOR")
    
    def export_cursor_set(self, set: int) -> bool:
        """If export scope was set to CURSOR, set the cursor set to be used
        Args:
            set: 1 or 2 for CURSOR1 or CURSOR2
        Returns:
            True if cursor set is not 1 or 2
        """
        if set not in (1,2):
            return True
        self.__write_to_dev(f"EXP:WAV:CURS {set}")
        return False
    
    """
    NOTICE
    Exporting multiple sources to a .zip file is not supported on firmware versions
    older than 2.3.2.2! Our model is currently on firmware 1.4.2.2.
    """
    
    def export_sources(self, *src: str) -> bool:
        """Select all waveforms to be exported to the file. Latest firmware (2.3.2.2) needed for multiple waveforms,
        else only the first source is selected.                                                                                                                                                                  
        Args:
            *src: One or more of the following waveforms
                Analogue - "C1","C2","C3","C4".
                Digital - "D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15".
                Math - "M1","M2","M3","M4","M5".
                Reference - "R1","R2","R3","R4".
                Spectrum - "SPECMAXH1","SPECMAXH2","SPECMAXH3","SPECMAXH4",
                    "SPECMINH1","SPECMINH2","SPECMINH3","SPECMINH4",
                    "SPECNORM1","SPECNORM2","SPECNORM3","SPECNORM4",
                    "SPECAVER1","SPECAVER2","SPECAVER3","SPECAVER4".
        Returns:
            True if any of the waveforms are invalid, no changes are applied in that case
        """
        if self._firmware_version >= "2.3.2.2" and all(self.is_source_active(x) for x in src):
            self.__write_to_dev(f"EXP:WAV:SOUR {','.join(src)}")
        elif self._firmware_version < "2.3.2.2" and self.is_source_active(src[0]):
            self.__write_to_dev(f"EXP:WAV:SOUR {src[0]}")
        else:
            return True
        return False
    
    def export_set_filename(self, filename: str) -> bool:
        """Set the filename for waveform exports. Local storage is in /home/storage/userData
        Args:
            filename: Path and filename with extension .csv or .ref for single waveform exports
        Returns:
            True if filename doesn't end on .csv or .ref, no other checks are done!
        """
        filename = filename.strip()
        if filename[-4:] not in (".csv", ".ref"): #, ".zip"):
            return True
        self.__write_to_dev(f"EXP:WAV:NAME '{filename}'")
        return False
    
    def export_get_filename(self) -> str:
        """Get the currently set filename for waveform exports
        Returns:
            Path and filename for waveform exports as a string
        """
        return self.__read_from_dev("EXP:WAV:NAME?")
    
    def fra_enter(self) -> None:
        """Enter frequency response analysis mode. This is done automatically whenever an FRA function is called.
        Returns:
            None
        """
        self.__write_to_dev("FRAN:ENAB ON")
        self.sync()
    
    def fra_exit(self):
        """Exit frequency response analysis mode
        Returns:
            None
        """
        self.__write_to_dev("FRAN:ENAB OFF")
        self.sync()
    
    def fra_freq_start(self, freq: float) -> None:
        """Set the start frequency of the sweep
        Args:
            freq: Frequency in Hz from 10 mHz to 100 MHz (value will be clamped)
        Returns:
            None
        """
        self.fra_enter()
        freq = self.__clamp(10*mHz, freq, 100*MHz)
        self.__write_to_dev(f"FRAN:FREQ:STAR {freq:.2f}")
    
    def fra_freq_stop(self, freq: float) -> None:
        """Set the stop frequency of the sweep
        Args:
            freq: Frequency in Hz from 10 mHz to 100 MHz (value will be clamped)
        Returns:
            None
        """
        self.fra_enter()
        freq = self.__clamp(10*mHz, freq, 100*MHz)
        self.__write_to_dev(f"FRAN:FREQ:STOP {freq:.2f}")
    
    def fra_run(self) -> None:
        """Run the frequency response analysis
        Returns:
            None
        """
        self.fra_enter()
        self.__write_to_dev("FRAN:STAT RUN")
    
    def fra_stop(self) -> None:
        """Stop the frequency response analysis
        Returns:
            None
        """
        self.fra_enter()
        self.__write_to_dev("FRAN:STAT STOP")
    
    def fra_generator(self, channel: int) -> bool:
        """Select built-in generator for a frequency sweep
        Args:
            channel: 1 or 2
        Returns:
            True for invalid channel number
        """
        if channel not in (1,2):
            return True
        self.fra_enter()
        self.__write_to_dev(f"FRAN:GEN GEN{channel}")
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
        self.__write_to_dev(f"FRAN:INP C{channel}")
        return False
    
    def fra_output_channel(self, channel: int) -> bool:
        """Set the channel used for the output signal of the device
        Args:
            channel: 1 to 4
        Returns:
            True for invalid channel number
        """
        if channel not in (1,2,3,4):
            return True
        self.fra_enter()
        self.__write_to_dev(f"FRAN:OUTP C{channel}")
        return False
    
    def fra_repeat(self, state: bool) -> None:
        """Whether to repeat the measurement using the same parameters
        Args:
            state: True to repeat
        Returns:
            None
        """
        self.fra_enter()
        self.__write_to_dev(f"FRAN:REP {int(state)}")
        
    
    def live_command_mode(self):
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

    def test(self, cmd):
        if '?' in cmd:
            return self.__read_from_dev(cmd)
        else:
            self.__write_to_dev(cmd)


if __name__ == "__main__":
    scan_instruments()

    d = DriverMXO4X()
    d.serial_start()
    d.get_id()

    d.do_reset()
    d.change_display_mode(True)
    d.change_remote_text("Hello World!")

    d.fra_freq_start(100)
    d.fra_run()
    d.sync()
    #d.live_command_mode()

    d.serial_close()
