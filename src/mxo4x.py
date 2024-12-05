import numpy as np
from time import sleep
import pyvisa
import platform
from RsInstrument import RsInstrument

KHz = 1000
MHz = 1000000

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

    def __init__(self):
        pass

    def __write_to_dev(self, order: str) -> None:
        """Wrapper for executing commands on device
        Args:
            order: command to run on device (may alter device state)
        Returns:
            None
        """
        self.SerialDevice.write(order)

    def __read_from_dev(self, order: str) -> str:
        """Wrapper for querying data from device
        Args:
            order: command to run on device
        Returns:
            Queried data as a string
        """
        text_out = self.SerialDevice.query(order)
        return text_out

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
        Args:
            N/A
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
        Args:
            N/A
        Returns:
            None
        """
        if not self.SerialActive:
            print("... not done due to wrong device")
        else:
            self.__write_to_dev("*RST")
            sleep(2)

    def change_display_mode(self, show_display: bool) -> None:
        """Decide whether display is shown during remote control
        Args:
            show_display: True to show display, False to show static image (may improve performance)
        Returns:
            None
        """
        self.__write_to_dev(f"SYST:DISP:UPD {int(show_display)}")
        sleep(2 if show_display else 1)

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
        Args:
            N/A
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
        Args:
            N/A
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
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:THC {int(state)}")

    def dig_show_bus(self, logic_group: int = None) -> None:
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:DISP:SHBU ON")

    def dig_hide_bus(self, logic_group: int = None) -> None:
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:DISP:SHBU OFF")

    def dig_bus_data_format(self, format: str, logic_group: int = None) -> bool:
        if (format := format.upper()) not in ("HEX", "OCT", "BIN", "ASCII", "SIGNED", "UNSIGNED"):
            return True
        if format == "SIGNED":
            format = "SIGN"
        elif format == "UNSIGNED":
            format = "USIG"
        logic_group = self.__fix_logic_index(logic_group)
        self.__write_to_dev(f"PBUS{logic_group}:DATA:FORM {format}")
        return False

    def dig_bus_header(self, logic_group: int = None) -> None:
        logic_group = self.__fix_logic_index(logic_group)
        return self.__read_from_dev(f"PBUS{logic_group}:DATA:HEAD?")

    def __fix_output_config(self, output_config: int) -> int:
        return output_config if output_config in (1,2) else self._output_config

    def sshot_get_default_config(self):
        return self._output_config

    def sshot_set_default_config(self, output_config: int) -> bool:
        if output_config in (1, 2):
            self._output_config = output_config
            return False
        return True

    def sshot_get_filename(self) -> str:
        return self.__read_from_dev("MMEM:NAME?")

    def sshot_set_filename(self, filename: str) -> None:
        self.__write_to_dev(f"MMEM:NAME {filename}")

    def sshot_destination(self, dest: str) -> bool:
        if dest.upper() == "FILE":
            self.__write_to_dev(f"MMEM:DEST MMEM")
        elif dest.upper() == "CLIPBOARD":
            self.__write_to_dev(f"MMEM:DEST CLIPBOARD")
        else:
            return True
        return False

    def sshot_file_format(self, format: str, output_config: int = None) -> bool:
        output_config = self.__fix_output_config(output_config)
        if format.upper() not in ("PNG", "JPG"):
            return True
        self.__write_to_dev(f"HCOP:DEV{output_config}:LANG {format.upper()}")
        return False

    def sshot_invert_colours(self, state: bool, output_config: int = None) -> None:
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:DEV{output_config}:INV {int(state)}")

    def sshot_white_background(self, state: bool, output_config: int = None) -> None:
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:DEV{output_config}:WBKG {int(state)}")

    def sshot_include_signal_bar(self, state: bool, output_config: int = None) -> None:
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:DEV{output_config}:ISBA {int(state)}")

    def sshot_include_dialog_box(self, state: bool, output_config: int = None) -> None:
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:DEV{output_config}:SSD {int(state)}")

    def sshot_capture_now(self, output_config: int = None) -> None:
        if int(self.__read_from_dev("SYST:DISP:UPD?")) == 0:
            self.change_display_mode(True)
        output_config = self.__fix_output_config(output_config)
        self.__write_to_dev(f"HCOP:IMM{output_config}")

    def sshot_capture_next(self, output_config: int = None) -> None:
        if int(self.__read_from_dev("SYST:DISP:UPD?")) == 0:
            self.change_display_mode(True)
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

    d.gen_preset()
    d.gen_enable()
    d.live_command_mode()


    d.serial_close()
