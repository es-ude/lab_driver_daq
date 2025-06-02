import numpy as np
from logging import getLogger
from tqdm import tqdm
from time import sleep
from datetime import datetime
from dataclasses import dataclass
from lab_driver.charac_common import CharacterizationCommon
from lab_driver.yaml_handler import YamlConfigHandler


@dataclass
class SettingsAmplifier:
    """Class with settings for testing an electrical amplifier stage (DUT)
    Attributes:
        vss:            Floating with minimal applied voltage
        vdd:            Floating with maximal applied voltage
        test_rang:      List with [min, max] analog ranges for amplifier (like: [-5.0, +5.0])
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

    def get_cycle_stimuli_input(self) -> np.ndarray:
        return np.linspace(start=self.test_rang[0], stop=self.test_rang[1], num=self.get_num_steps(), endpoint=True, dtype=float)

    def get_cycle_empty_array(self) -> np.ndarray:
        """Function for generating an empty numpy array with right size"""
        assert self.daq_ovr > 0, "Variable: daq_ovr - Must be greater than 1"
        return np.zeros(shape=(self.num_rpt, self.get_cycle_stimuli_input().size, self.daq_ovr), dtype=float)


DefaultSettingsAmplifier = SettingsAmplifier(
    vss=0.0,
    vdd=5.0,
    test_rang=[0.0, 5.0],
    daq_ovr=1,
    num_rpt=1,
    delta_steps=0.05,
    sleep_sec=0.1
)


class CharacterizationAmplifier(CharacterizationCommon):
    _input_val: float
    settings: SettingsAmplifier

    def __init__(self, folder_reference: str) -> None:
        """Class for handling the measurement routine for characterising a Digital-Analog-Converter (DAC)
        :param folder_reference:    String with source folder to find the Main Repo Path
        """
        super().__init__()
        self._logger = getLogger(__name__)
        self.settings = YamlConfigHandler(
            yaml_template=DefaultSettingsAmplifier,
            path2yaml='config',
            yaml_name='Config_TestAmplifier',
            folder_reference=folder_reference,
        ).get_class(SettingsAmplifier)

    def run_test_transfer(self, chnl: int, func_mux, func_set_daq, func_sens, func_get_daq, func_beep) -> dict:
        """Function for characterizing the transfer function of the Amplifier
        :param chnl:            Integer number for testing the channel
        :param func_mux:        Function for defining the testmode of the hardware DUT with defining the channel (chnl)
        :param func_set_daq:    Function for applying selected voltage/current signal on DAQ with input params (data)
        :param func_sens:       Function for getting the applied voltage/current (sensing) from DAQ device
        :param func_get_daq:    Function for sensing the DUT-amplifier output from DAQ
        :param func_beep:       Function for do a beep in DAQ
        :return:                Dictionary with ['stim': input test signal of one repetition, 'settings': Settings, 'rpt<X>': Test results]"""
        stimuli = self.settings.get_cycle_stimuli_input()
        results = {'stim': stimuli}

        sens_test = self.settings.get_cycle_empty_array()
        results_ch = self.settings.get_cycle_empty_array()
        func_mux(chnl)
        self._logger.debug(f"Prepared DAC channel: {chnl}")

        for rpt_idx in range(self.settings.num_rpt):
            for val_idx, data in enumerate(tqdm(stimuli, ncols=100, desc=f"Process CH{chnl} @ repetition {1 + rpt_idx}/{self.settings.num_rpt}")):
                self._input_val = data
                func_set_daq(data)
                sleep(self.settings.sleep_sec)
                for daq_idx in range(self.settings.daq_ovr):
                    sens_test[rpt_idx, val_idx, daq_idx] = func_sens()
                    results_ch[rpt_idx, val_idx, daq_idx] = func_get_daq()

            func_beep()
            results.update({f"rpt{rpt_idx:02d}": results_ch})
        return results
