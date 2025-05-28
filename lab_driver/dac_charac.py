import numpy as np
from logging import getLogger
from os import makedirs
from os.path import join
from tqdm import tqdm
from time import sleep
from random import random
from datetime import datetime
from dataclasses import dataclass
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
    """
    dac_reso: int
    dac_chnl: list
    dac_rang: list
    daq_ovr: int
    num_rpt: int
    num_steps: int

    @staticmethod
    def get_date_string() -> str:
        """Function for getting the datetime in string format"""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_cycle_stimuli_input(self) -> np.ndarray:
        num_points = int((self.dac_rang[1] - self.dac_rang[0] + 1) / self.num_steps)
        return np.linspace(start=self.dac_rang[0], stop=self.dac_rang[1], num=num_points, endpoint=True, dtype=int)

    def get_cycle_empty_array(self) -> np.ndarray:
        """Function for generating an empty numpy array with right size"""
        return np.zeros(shape=(self.num_rpt, self.get_cycle_stimuli_input().size, self.daq_ovr), dtype=float)


DefaultSettingsDAC = SettingsDAC(
    dac_reso=16,
    dac_chnl=[idx for idx in range(16)],
    dac_rang=[0, 2**16-1],
    daq_ovr=1,
    num_rpt=1,
    num_steps=1,
)


class CharacterizationDAC:
    _sleep_set_sec: float = 0.01
    settings: SettingsDAC

    def __init__(self) -> None:
        self._logger = getLogger(__name__)
        self.settings = YamlConfigHandler(
            yaml_template=DefaultSettingsDAC,
            path2yaml='config',
            yaml_name='Config_TestDAC'
        ).get_class(SettingsDAC)
        self.check_settings_error()

    def check_settings_error(self) -> None:
        """Function for checking for input errors"""
        assert self.settings.daq_ovr > 0, "Variable: daq_ovr - Must be greater than 1"
        assert len(self.settings.dac_rang) == 2, "Variable: dac_rang - Length must be 2"

    @staticmethod
    def bypass_mock_beep() -> None:
        """Function for bypassing the beep on DAQ device
        :return:        None
        """
        pass

    @staticmethod
    def bypass_mock_mux(chnl: int) -> None:
        """Function for bypassing the definition of the multiplexer stage
        :param chnl:    Integer with MUX number
        :return:        None
        """
        pass

    @staticmethod
    def bypass_mock_dac(chnl: int, data: int) -> None:
        """Function for bypassing the definition of the DAC output
        :param chnl:    Integer with DAC number
        :param data:    Integer with DAC data
        :return:        None
        """
        pass

    @staticmethod
    def bypass_mock_daq() -> float:
        """Function for bypassing the definition of the DAQ measurement device
        :return:        Floating value in range of [-1, +1]
        """
        return 2*(0.5 - random())

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
                    sleep(self._sleep_set_sec)
                    for ovr_idx in range(self.settings.daq_ovr):
                        results_ch[rpt_idx, val_idx, ovr_idx] = func_daq()
                self._logger.debug(f"Sweep DAC channel: {chnl}")

                func_beep()
            results.update({f"ch{chnl:02d}": results_ch})
        return results

    def save_results(self, data: dict, folder_name: str) -> None:
        """Function for saving the measured data in numpy format
        :param data:        Dictionary with results from measurement
        :param folder_name: Name of folder where results will be saved
        :return:            None
        """
        makedirs(folder_name, exist_ok=True)
        np.savez_compressed(
            file=join(folder_name, 'dac_measured_data.npz'),
            allow_pickle=True,
            data=data,
            settings=self.settings,
            date=self.settings.get_date_string()
        )
        self._logger.debug(f"Saved results in folder: {folder_name}")
        self._logger.debug(f"Saved measured with {len(data)} entries")
