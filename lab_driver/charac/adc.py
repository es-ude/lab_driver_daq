import numpy as np
from os.path import splitext
from logging import getLogger, Logger
from tqdm import tqdm
from time import sleep
from datetime import datetime
from dataclasses import dataclass
from lab_driver.yaml_handler import YamlConfigHandler
from lab_driver.charac.common import CharacterizationCommon
from lab_driver.process.data import MetricCalculator
from lab_driver.plots import plot_transfer_function_norm, plot_transfer_function_metric


@dataclass
class SettingsADC:
    """Class with settings for testing the DUT of Analog-Digital-Converter (ADC)
    Attributes:
        system_id:      String with system name or ID
        voltage_min:    Floating with minimal ADC voltage
        voltage_max:    Floating with maximal ADC voltage
        adc_reso:       Integer with bit resolution of ADC
        adc_chnl:       List with ADC channel IDs to test (like: [idx for idx in range (16)])
        adc_rang:       List with [min, max] analog ranges for testing the N-bit ADC (like: [0, 65535])
        daq_ovr:        Integer number for oversampling of DAQ system
        num_rpt:        Integer of completes cycles to run DAQ
        sleep_sec:      Sleeping seconds between each DAQ setting
    """
    system_id: str
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
    system_id='0',
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
        for _ in range(4):
            sleep(0.5)
            func_beep()

        return results

    def plot_characteristic_results_from_file(self, path: str, file_name: str) -> None:
        """Function for plotting the loaded data files
        :param path:        Path to the numpy files with DAQ results
        :param file_name:   Name of numpy array with DAQ results to load
        :return:            None
        """
        self._logger.info('Loading the data file')
        data = MetricCalculator().load_data(
            path=path,
            file_name=file_name
        )['data']
        self._logger.info('Calculating the metric')

        self.__plot_characteristic(
            data=data,
            path2save=path,
            file_name=file_name
        )

    def plot_characteristic_results_direct(self, data: dict, file_name: str, path: str) -> None:
        """Function for plotting the loaded data files
        :param data:        Dictionary with measurement data ['stim', 'ch<x>', ...]
        :param path:        Path to measurement in which the figures are saved
        :param file_name:   Name of figure file to save
        :return:            None
        """
        hndl = MetricCalculator()
        self._logger.info('Calculating the metric')
        metric = hndl.process_data_direct(data)
        self.__plot_characteristic(
            data=metric,
            path2save=path,
            file_name=file_name
        )

    def __plot_characteristic(self, data: dict, path2save: str, file_name: str) -> None:
        self._logger.info('Plotting the signals')
        hndl = MetricCalculator()
        file_name_wo_ext = splitext(file_name)[0]

        xtext = r'Voltage $V_{in}$ [V]'
        plot_transfer_function_norm(
            data=data,
            path2save=path2save,
            xlabel=xtext,
            ylabel='ADC Output',
            title='',
            file_name=f"{file_name_wo_ext}_norm"
        )
        plot_transfer_function_metric(
            data=data,
            func=hndl.calculate_lsb,
            path2save=path2save,
            xlabel=xtext,
            ylabel='ADC LSB [V]',
            title='',
            file_name=f"{file_name_wo_ext}_lsb"
        )
        plot_transfer_function_metric(
            data=data,
            func=hndl.calculate_dnl,
            path2save=path2save,
            xlabel=xtext,
            ylabel='ADC DNL',
            title='',
            file_name=f"{file_name_wo_ext}_dnl"
        )
