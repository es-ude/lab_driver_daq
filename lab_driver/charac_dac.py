import numpy as np
from logging import getLogger
from tqdm import tqdm
from time import sleep
from datetime import datetime
from dataclasses import dataclass
from lab_driver.charac_common import CharacterizationCommon
from lab_driver.yaml_handler import YamlConfigHandler


@dataclass
class SettingsDAC:
    """Class with settings for testing the DUT of Digital-Analog-Converter (DAC)
    Attributes:
        dac_reso:   Integer with bit resolution of DAC
        dac_chnl:   List with DAC channel IDs to test (like: [idx for idx in range (16)])
        dac_rang:   List with [min, max] digital ranges for testing the N-bit DAC (like: [0, 65535])
        daq_ovr:    Integer number for oversampling of DAQ system
        num_rpt:    Integer of completes cycles to run DAQ
        num_steps:  Integer of intermediate steps in ramping
        sleep_sec:      Sleeping seconds between each DAQ setting
    """
    dac_reso: int
    dac_chnl: list
    dac_rang: list
    daq_ovr: int
    num_rpt: int
    num_steps: int
    sleep_sec: float

    @staticmethod
    def get_date_string() -> str:
        """Function for getting the datetime in string format"""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_num_steps(self) -> int:
        """Function for getting the number of steps in testing"""
        assert len(self.dac_rang) == 2, "Variable: adc_rang - Length must be 2"
        assert self.dac_rang[0] < self.dac_rang[1], "Variable: adc_rang[0] must be smaller than adc_rang[1]"
        return int((self.dac_rang[1] - self.dac_rang[0]) / self.num_steps) + 1

    def get_cycle_stimuli_input(self) -> np.ndarray:
        """Getting the numpy array with a stimuli with sawtooth waveform"""
        return np.linspace(start=self.dac_rang[0], stop=self.dac_rang[1], num=self.get_num_steps(), endpoint=True, dtype=int)

    def get_cycle_empty_array(self) -> np.ndarray:
        """Function for generating an empty numpy array with right size"""
        assert self.daq_ovr > 0, "Variable: adc_ovr - Must be greater than 1"
        return np.zeros(shape=(self.num_rpt, self.get_num_steps(), self.daq_ovr), dtype=float)


DefaultSettingsDAC = SettingsDAC(
    dac_reso=16,
    dac_chnl=[idx for idx in range(16)],
    dac_rang=[0, 2**16-1],
    daq_ovr=1,
    num_rpt=1,
    num_steps=1,
    sleep_sec=0.1
)


class CharacterizationDAC(CharacterizationCommon):
    settings: SettingsDAC

    def __init__(self, folder_reference: str) -> None:
        """Class for handling the measurement routine for characterising a Digital-Analog-Converter (DAC)
        :param folder_reference:    String with source folder to find the Main Repo Path
        """
        super().__init__()
        self._logger = getLogger(__name__)
        self.settings = YamlConfigHandler(
            yaml_template=DefaultSettingsDAC,
            path2yaml='config',
            yaml_name='Config_TestDAC',
            folder_reference=folder_reference
        ).get_class(SettingsDAC)

    def run_test_dac_transfer(self, func_mux, func_dac, func_daq, func_beep) -> dict:
        """Function for characterizing the transfer function of the DAC
        :param func_mux:    Function for defining the pre-processing part of the hardware DUT, setting the DAC channel with inputs (chnl)
        :param func_dac:    Function for applying selected channel and data on DUT-DAC with input params (chnl, data)
        :param func_daq:    Function for sensing the DAC output with external multimeter device
        :param func_beep:   Function for do a beep in DAQ
        :return:            Dictionary with ['stim': DAC input stream, 'settings': Settings, 'ch<X>': DAQ results with 'val' and 'std']"""
        stimuli = self.settings.get_cycle_stimuli_input()

        results = {'stim': stimuli}
        for chnl in self.settings.dac_chnl:
            results_ch = self.settings.get_cycle_empty_array()
            func_mux(chnl)
            self._logger.debug(f"Prepared DAC channel: {chnl}")

            for rpt_idx in range(self.settings.num_rpt):
                for val_idx, data in enumerate(tqdm(stimuli, ncols=100, desc=f"Process CH{chnl} @ repetition {1 + rpt_idx}/{self.settings.num_rpt}")):
                    func_dac(chnl, data)
                    sleep(self.settings.sleep_sec)
                    for ovr_idx in range(self.settings.daq_ovr):
                        results_ch[rpt_idx, val_idx, ovr_idx] = func_daq()
                self._logger.debug(f"Sweep DAC channel: {chnl}")

                func_beep()
            results.update({f"ch{chnl:02d}": results_ch})
        return results
