import numpy as np
from logging import getLogger, Logger
from tqdm import tqdm
from time import sleep
from datetime import datetime
from dataclasses import dataclass

from lab_driver.charac_common import CharacterizationCommon
from lab_driver.yaml_handler import YamlConfigHandler


@dataclass
class SettingsDevice:
    """Class with settings for testing the electrical device (DUT)
    Attributes:
        vss:            Floating with minimal applied voltage
        vdd:            Floating with maximal applied voltage
        test_rang:      List with [min, max] analog ranges for testing the N-bit ADC (like: [0, 65535])
        daq_ovr:        Integer number for oversampling of DAQ system
        num_rpt:        Integer of completes cycles to run DAQ
        delta_steps:    Float of intermediate steps in changing values
        sleep_sec:      Sleeping seconds between each DAQ setting
    """
    vss: float
    vdd: float
    test_rang: list
    daq_ovr: int
    num_rpt: int
    delta_steps: float
    sleep_sec: float

    @staticmethod
    def get_date_string() -> str:
        """Function for getting the datetime in string format"""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_common_mode_voltage(self) -> float:
        """Getting the Common Mode Voltage (VCM) of the DUT"""
        return (self.vdd + self.vss) / 2

    def get_num_steps(self) -> int:
        """Function for getting the number of steps in testing"""
        assert len(self.test_rang) == 2, "Variable: test_rang - Length must be 2"
        assert self.test_rang[0] < self.test_rang[1], "Variable: test_rang[0] must be smaller than test_rang[1]"
        assert self.vss <= self.test_rang[0] and self.vss < self.test_rang[1], "Variable: vss must be smaller than test_rang"
        assert self.vdd > self.test_rang[0] and self.vdd >= self.test_rang[1], "Variable: vdd must be greater than test_rang"
        return int((self.test_rang[1] - self.test_rang[0]) / self.delta_steps) + 1

    def get_cycle_stimuli_input_sawtooth(self) -> np.ndarray:
        """Getting the numpy array with a stimuli with sawtooth waveform"""
        return np.linspace(start=self.test_rang[0], stop=self.test_rang[1], num=self.get_num_steps(), endpoint=True, dtype=float)

    def get_cycle_stimuli_input_sinusoidal(self) -> np.ndarray:
        """Getting the numpy array with a stimuli input with sinusoidal waveform"""
        time = np.linspace(start=0.0, stop=2 * np.pi, num=2+self.get_num_steps(), endpoint=True, dtype=float)
        vcm = (self.test_rang[1] + self.test_rang[0]) / 2
        return vcm + (self.test_rang[1] - vcm) * np.sin(time)

    def get_cycle_stimuli_input_triangular(self) -> np.ndarray:
        """Getting the numpy array with a stimuli input with triangular waveform"""
        ramp = np.linspace(start=0.0, stop=1.0, num=1+int(np.ceil((self.get_num_steps()-1)/4)), endpoint=True, dtype=float)
        sig = np.concatenate((ramp, np.flip(ramp[:-1]), -ramp[1:], -np.flip(ramp[:-1])), axis=0)
        vcm = (self.test_rang[1] + self.test_rang[0]) / 2
        return vcm + (self.test_rang[1] - vcm) * sig

    def get_cycle_empty_array(self) -> np.ndarray:
        """Function for generating an empty numpy array with right size"""
        assert self.daq_ovr > 0, "Variable: daq_ovr - Must be greater than 1"
        return np.zeros(shape=(self.num_rpt, self.get_num_steps(), self.daq_ovr), dtype=float)


DefaultSettingsDevice = SettingsDevice(
    vss=0.0,
    vdd=5.0,
    test_rang=[0.0, 5.0],
    daq_ovr=1,
    num_rpt=1,
    delta_steps=0.05,
    sleep_sec=0.1
)


class CharacterizationDevice(CharacterizationCommon):
    settings: SettingsDevice
    _input_val: float
    _logger: Logger

    def __init__(self, folder_reference: str) -> None:
        """Class for handling the measurement routine for characterising an electrical device
        :param folder_reference:    String with source folder to find the Main Repo Path
        """
        super().__init__()
        self._logger = getLogger(__name__)
        self.settings = YamlConfigHandler(
            yaml_template=DefaultSettingsDevice,
            path2yaml='config',
            yaml_name='Config_TestDevice',
            folder_reference=folder_reference,
        ).get_class(SettingsDevice)

    def run_test_transfer(self, func_stim, func_daq, func_sens, func_resp, func_beep) -> dict:
        """Function for characterizing the transfer function of the ADC
        :param func_stim:   Function to build the test signal (sawtooth, sinusoidal, triangular, pulse) from settings
        :param func_daq:    Function for setting the voltage/current value on DAQ with input params (data)
        :param func_sens:   Function for sensing the applied voltage/current (measured) from DAQ device
        :param func_resp:    Function for getting the applied voltage/current (output) from DAQ device
        :param func_beep:   Function for do a beep in DAQ
        :return:            Dictionary with ['stim': input test signal of one repetition, 'settings': Settings, 'rpt<X>': Test results]"""
        stimuli = func_stim()
        results = {'stim': stimuli}

        sens_test = self.settings.get_cycle_empty_array()
        results_ch = self.settings.get_cycle_empty_array()

        for rpt_idx in range(self.settings.num_rpt):
            for val_idx, data in enumerate(tqdm(stimuli, ncols=100, desc=f"Process repetition {1 + rpt_idx}/{self.settings.num_rpt}")):
                self._input_val = data
                func_daq(data)

                sleep(self.settings.sleep_sec)
                for daq_idx in range(self.settings.daq_ovr):
                    sens_test[rpt_idx, val_idx, daq_idx] = func_sens()
                    results_ch[rpt_idx, val_idx, daq_idx] = func_resp()

            func_beep()
            results.update({f"rpt{rpt_idx:02d}": results_ch})
        for _ in range(4):
            sleep(0.5)
            func_beep()

        return results
