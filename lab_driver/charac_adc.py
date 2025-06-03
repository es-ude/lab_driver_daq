import numpy as np
from logging import getLogger, Logger
from tqdm import tqdm
from time import sleep
from datetime import datetime
from dataclasses import dataclass
from lab_driver.charac_common import CharacterizationCommon
from lab_driver.yaml_handler import YamlConfigHandler


@dataclass
class SettingsADC:
    """Class with settings for testing the DUT of Analog-Digital-Converter (ADC)
    Attributes:
        voltage_min:    Floating with minimal ADC voltage
        voltage_max:    Floating with maximal ADC voltage
        adc_reso:       Integer with bit resolution of ADC
        adc_chnl:       List with ADC channel IDs to test (like: [idx for idx in range (16)])
        adc_rang:       List with [min, max] analog ranges for testing the N-bit ADC (like: [0, 65535])
        daq_ovr:        Integer number for oversampling of DAQ system
        num_rpt:        Integer of completes cycles to run DAQ
        sleep_sec:      Sleeping seconds between each DAQ setting
    """
    voltage_min: float
    voltage_max: float
    adc_reso: int
    adc_chnl: list
    adc_rang: list
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
        return (self.voltage_min + self.voltage_max) / 2

    def get_num_steps(self) -> int:
        """Function for getting the number of steps in testing"""
        assert len(self.adc_rang) == 2, "Variable: adc_rang - Length must be 2"
        assert self.adc_rang[0] < self.adc_rang[1], "Variable: adc_rang[0] must be smaller than adc_rang[1]"
        assert self.voltage_min <= self.adc_rang[0] and self.voltage_min < self.adc_rang[1], "Variable: voltage_min must be smaller than adc_rang"
        assert self.voltage_max > self.adc_rang[0] and self.voltage_max >= self.adc_rang[1], "Variable: voltage_max must be greater than adc_rang"
        return int((self.adc_rang[1] - self.adc_rang[0]) / self.delta_steps) + 1

    def get_cycle_stimuli_input(self) -> np.ndarray:
        """Getting the numpy array with a stimuli with sawtooth waveform"""
        return np.linspace(start=self.adc_rang[0], stop=self.adc_rang[1], num=self.get_num_steps(), endpoint=True, dtype=float)

    def get_cycle_empty_array(self) -> np.ndarray:
        """Function for generating an empty numpy array with right size"""
        assert self.daq_ovr > 0, "Variable: daq_ovr - Must be greater than 1"
        return np.zeros(shape=(self.num_rpt, self.get_num_steps(), self.daq_ovr), dtype=float)


DefaultSettingsADC = SettingsADC(
    voltage_min=0.0,
    voltage_max=5.0,
    adc_reso=16,
    adc_chnl=[idx for idx in range(16)],
    adc_rang=[0.0, 5.0],
    daq_ovr=1,
    num_rpt=1,
    delta_steps=0.05,
    sleep_sec=0.1
)


class CharacterizationADC(CharacterizationCommon):
    settings: SettingsADC
    _logger: Logger
    _input_val: float

    def __init__(self, folder_reference: str) -> None:
        """Class for handling the measurement routine for characterising a Digital-Analog-Converter (DAC)
        :param folder_reference:    String with source folder to find the Main Repo Path
        """
        super().__init__()
        self._logger = getLogger(__name__)
        self.settings = YamlConfigHandler(
            yaml_template=DefaultSettingsADC,
            path2yaml='config',
            yaml_name='Config_TestADC',
            folder_reference=folder_reference,
        ).get_class(SettingsADC)

    def run_test_transfer(self, func_mux, func_daq, func_sens, func_dut, func_beep) -> dict:
        """Function for characterizing the transfer function of the ADC
        :param func_mux:    Function for defining the pre-processing part of the hardware DUT, setting the ADC channel with inputs (chnl)
        :param func_daq:    Function for applying selected channel and data on DAQ with input params (data)
        :param func_sens:   Function for getting the applied voltage (measured) from DAQ device
        :param func_dut:    Function for sensing the ADC output with external multimeter device with inputs (chnl)
        :param func_beep:   Function for do a beep in DAQ
        :return:            Dictionary with ['stim': input test signal of one repetition, 'settings': Settings, 'ch<X>': DUT results with 'val' and 'std']"""
        stimuli = self.settings.get_cycle_stimuli_input()

        results = {'stim': stimuli}
        for chnl in self.settings.adc_chnl:
            sens_test = self.settings.get_cycle_empty_array()
            results_ch = self.settings.get_cycle_empty_array()
            func_mux(chnl)
            self._logger.debug(f"Prepared ADC channel: {chnl}")

            for rpt_idx in range(self.settings.num_rpt):
                for val_idx, data in enumerate(tqdm(stimuli, ncols=100, desc=f"Process CH{chnl} @ repetition {1 + rpt_idx}/{self.settings.num_rpt}")):
                    self._input_val = data
                    func_daq(data)
                    sleep(self.settings.sleep_sec)
                    for daq_idx in range(self.settings.daq_ovr):
                        sens_test[rpt_idx, val_idx, daq_idx] = func_sens()
                        results_ch[rpt_idx, val_idx, daq_idx] = func_dut(chnl)
                self._logger.debug(f"Sweep ADC channel: {chnl}")

                func_beep()
            results.update({f"ch{chnl:02d}": results_ch})
        return results
