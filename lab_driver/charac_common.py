from random import random, randint


class CharacterizationCommon:
    _sleep_set_sec: float = 0.01

    def set_sleeping_setting_input(self, sleep_sec: float) -> None:
        """Function for setting the sleeping seconds between setting the input stimuli and sensing output
        :param sleep_sec:   Floating with sleeping seconds
        :return:            None
        """
        self._sleep_set_sec = abs(sleep_sec)

    def get_sleeping_setting_input(self) -> float:
        """Function for setting the sleeping seconds between setting the input stimuli and sensing output
        :return:            Floating with sleeping seconds
        """
        return self._sleep_set_sec

    @staticmethod
    def dummy_beep() -> None:
        """Function for bypassing the beep on DAQ device
        :return:        None
        """
        pass

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
