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
class SettingsADC:
    """Class with settings for testing the DUT of Analog-Digital-Converter (ADC)
    Attributes:
        adc_voltage_min:    Floating with minimal ADC voltage
        adc_voltage_max:    Floating with maximal ADC voltage
        adc_reso:   Integer with bit resolution of ADC
        adc_chnl:   List with ADC channel IDs to test (like: [idx for idx in range (16)])
        adc_rang:   List with [min, max] analog ranges for testing the N-bit ADC (like: [0, 65535])
        adc_ovr:    Integer number for oversampling of DAQ system
        num_rpt:    Integer of completes cycles to run DAQ
        delta_steps:  Float of intermediate steps in ramping
    """
    adc_voltage_min: float
    adc_voltage_max: float
    adc_reso: int
    adc_chnl: list
    adc_rang: list
    adc_ovr: int
    num_rpt: int
    delta_steps: float

    @staticmethod
    def get_date_string() -> str:
        """Function for getting the datetime in string format"""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_cycle_stimuli_input(self) -> np.ndarray:
        num_points = (self.adc_rang[1] - self.adc_rang[0]) / self.delta_steps
        return np.linspace(start=self.adc_rang[0], stop=self.adc_rang[1], num=num_points, endpoint=True, dtype=float)

    def get_cycle_empty_array(self) -> np.ndarray:
        """Function for generating an empty numpy array with right size"""
        return np.zeros(shape=(self.num_rpt, self.get_cycle_stimuli_input().size), dtype=float)


DefaultSettingsADC = SettingsADC(
    adc_voltage_min=0.0,
    adc_voltage_max=5.0,
    adc_reso=16,
    adc_chnl=[idx for idx in range(16)],
    adc_rang=[0.0, 5.0],
    adc_ovr=1,
    num_rpt=1,
    delta_steps=0.05,
)


class CharacterizationADC:
    _sleep_set_sec: float = 0.01
    settings: SettingsADC

    def __init__(self, folder_reference: str) -> None:
        """Class for handling the measurement routine for characterising a Digital-Analog-Converter (DAC)
        :param folder_reference:    String with source folder to find the Main Repo Path
        """
        self._logger = getLogger(__name__)
        self.settings = YamlConfigHandler(
            yaml_template=DefaultSettingsADC,
            path2yaml='config',
            yaml_name='Config_TestADC',
            folder_reference=folder_reference,
        ).get_class(SettingsADC)
        self.check_settings_error()

    def check_settings_error(self) -> None:
        """Function for checking for input errors"""
        assert self.settings.adc_ovr > 0, "Variable: adc_ovr - Must be greater than 1"
        assert len(self.settings.adc_rang) == 2, "Variable: adc_rang - Length must be 2"

        assert self.settings.adc_rang[0] < self.settings.adc_rang[1], "Variable: adc_rang[0] must be smaller than adc_rang[1]"
        assert self.settings.adc_voltage_min <= self.settings.adc_rang[0] and self.settings.adc_voltage_min < self.settings.adc_rang[1], "Variable: adc_voltage_min must be smaller than adc_rang"
        assert self.settings.adc_voltage_max > self.settings.adc_rang[0] and self.settings.adc_voltage_max >= self.settings.adc_rang[1], "Variable: adc_voltage_max must be greater than adc_rang"

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
        for chnl in self.settings.adc_chnl:
            results_ch_val = self.settings.get_cycle_empty_array()
            results_ch_std = self.settings.get_cycle_empty_array()
            func_mux(chnl)
            self._logger.debug(f"Prepared DAC channel: {chnl}")

            for rpt_idx in range(self.settings.num_rpt):
                for val_idx, data in enumerate(tqdm(stimuli, ncols=100, desc=f"Process CH{chnl} @ repetition {1 + rpt_idx}/{self.settings.num_rpt}")):
                    func_dac(chnl, data)
                    sleep(self._sleep_set_sec)
                    read_volt = np.array([func_daq() for _ in range(self.settings.adc_ovr)])
                    results_ch_val[rpt_idx, val_idx] = np.mean(read_volt)
                    results_ch_std[rpt_idx, val_idx] = np.std(read_volt)
                self._logger.debug(f"Sweep DAC channel: {chnl}")

                func_beep()
            results.update({f"ch{chnl:02d}": {'val': results_ch_val, 'std': results_ch_std}})
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
