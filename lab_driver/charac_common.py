import numpy as np
from logging import getLogger
from os import makedirs
from os.path import join
from random import random, randint


class CharacterizationCommon:
    _input_val: float | int

    def __init__(self) -> None:
        """Common class with functions used in all characterisation methods"""
        self._logger = getLogger(__name__)

    @staticmethod
    def dummy_beep() -> None:
        """Function for bypassing the beep on DAQ device
        :return:        None
        """
        pass

    @staticmethod
    def dummy_reset() -> None:
        """Function for bypassing the device reset
        :return:        None
        """
        pass

    def dummy_get_stim_value(self) -> float | int:
        """Function for getting the input stimulation value"""
        return self._input_val

    @staticmethod
    def dummy_set_mux(chnl: int) -> None:
        """Function for bypassing the definition of the multiplexer stage
        :param chnl:    Integer with MUX number
        :return:        None
        """
        pass

    @staticmethod
    def dummy_set_dut_dac(chnl: int, data: int) -> None:
        """Function for bypassing the definition of the DAC output
        :param chnl:    Integer with DAC number
        :param data:    Integer with DAC data
        :return:        None
        """
        pass

    @staticmethod
    def dummy_get_dut_adc(chnl: int) -> int:
        """Function for bypassing the definition of the ADC output
        :param chnl:    Integer with DAC number
        :param data:    Integer with DAC data
        :return:        None
        """
        return randint(a=0, b=(2**16)-1)

    @staticmethod
    def dummy_get_daq() -> float:
        """Function for bypassing the definition of the DAQ measurement device
        :return:        Floating value in range of [-1, +1]
        """
        return 2 * (0.5 - random())

    @staticmethod
    def dummy_set_daq(val: float) -> None:
        """Function for bypassing the definition of the DAQ measurement device
        :return:        Floating value in range of [-1, +1]
        """
        pass

    def save_results(self, file_name: str, settings: object, data: dict, folder_name: str) -> None:
        """Function for saving the measured data in numpy format
        :param file_name:   Name of file to save (without extension)
        :param settings:    Class with settings
        :param data:        Dictionary with results from measurement
        :param folder_name: Name of folder where results will be saved
        :return:            None
        """
        makedirs(folder_name, exist_ok=True)
        np.savez_compressed(
            file=join(folder_name, f'{file_name}.npz'),
            allow_pickle=True,
            data=data,
            settings=settings
        )
        self._logger.debug(f"Saved results in folder: {folder_name}")
        self._logger.debug(f"Saved measured with {len(data)} entries")
