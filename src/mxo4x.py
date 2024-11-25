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
        for value in functions.values():
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

    def dig_threshold(self, threshold: float, channel_group: int, logic_group: int = None) -> bool:
        """Manually set a logical threshold voltage for some channel group
        Args:
            threshold: threshold voltage in volt in range [-8,+8]
            channel_group: 1 = digital channels 0..3
                2 = digital channels 4..7
                3 = digital channels 8..11
                4 = digital channels 12..15
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

    def live_command_mode(self):
        print("----- LIVE COMMAND MODE -----")
        print("---- type 'exit' to stop ----")
        while "exit" not in (cmd := input()):
            try:
                exec(cmd)
            except:
                print("- Command failed. Try again.")
        print("----- END OF LIVE COMMAND MODE -----")

    def test(self, cmd):
        if '?' in cmd:
            print(self.__read_from_dev(cmd))
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
