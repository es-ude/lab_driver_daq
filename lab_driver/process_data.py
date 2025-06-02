import numpy as np
from logging import getLogger
from lab_driver.process_common import ProcessCommon


class ProcessTransferFunction(ProcessCommon):
    def __init__(self) -> None:
        """Class with constructors for processing the measurement results for extracting device-specific metrics"""
        super().__init__()
        self._logger = getLogger(__name__)

    def calculate_lsb_mean(self, stim_input: np.ndarray, daq_output: np.ndarray) -> float:
        """Function for calculating the mean Least Significant Bit (LSB)
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output
        :return:        Float with LSB
        """
        return float(np.mean(self.calculate_lsb(
            stim_input=stim_input,
            daq_output=daq_output
        ), axis=0))

    @staticmethod
    def calculate_lsb(stim_input: np.ndarray, daq_output: np.ndarray) -> np.array:
        """Function for calculating the mean Least Significant Bit (LSB)
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output
        :return:        Float with LSB
        """
        assert daq_output.shape == stim_input.shape, "Dimension / shape mismatch"
        return np.diff(daq_output) / np.diff(stim_input)

    def calculate_dnl(self, stim_input: np.ndarray, daq_output: np.ndarray) -> np.ndarray:
        """Calculating the Differential Non-Linearity (DNL) of a transfer function from DAC/ADC
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output
        :return:            Numpy array with DNL
        """
        return self.calculate_lsb(stim_input, daq_output) - 1

    @staticmethod
    def calculate_inl(stim_input: np.ndarray, daq_output: np.ndarray) -> np.ndarray:
        """Calculating the Integral Non-Linearity (INL) of a transfer function from DAC/ADC
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output
        :return:        Numpy array with INL
        """
        raise NotImplementedError

