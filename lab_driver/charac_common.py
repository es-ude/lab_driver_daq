from random import random


class CharacterizationCommon:
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
        return 2 * (0.5 - random())
