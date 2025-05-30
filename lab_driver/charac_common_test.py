import unittest
import numpy as np
from lab_driver.charac_common import CharacterizationCommon


class TestCommon(unittest.TestCase):
    def test_common_charac_func_random_voltage_float(self):
        hndl = CharacterizationCommon()
        result = hndl.dummy_get_daq()
        self.assertTrue(type(result) == float)

    def test_common_charac_func_random_voltage_array(self):
        hndl = CharacterizationCommon()
        result = np.array([hndl.dummy_get_daq() for _ in range(100)])
        check = result.shape == (100, ) and result.min() >= -1.0 and result.max() <= 1.0
        self.assertTrue(check)

    def test_common_charac_func_dut_adc(self):
        hndl = CharacterizationCommon()
        result = np.array([hndl.dummy_get_dut_adc(0) for _ in range(100)])
        check = result.shape == (100,) and result.min() >= 0.0 and result.max() <= 2**16-1
        self.assertTrue(check)

    def test_common_sleeping_value(self):
        hndl = CharacterizationCommon()
        test_sleep = [-1.0, -0.5, 0.0, 0.5, 1.0]
        chck_sleep = [1.0, 0.5, 0.0, 0.5, 1.0]

        result_sleep = list()
        for val in test_sleep:
            hndl.set_sleeping_setting_input(val)
            result_sleep.append(hndl.get_sleeping_setting_input())

        np.testing.assert_array_equal(result_sleep, chck_sleep)

if __name__ == '__main__':
    unittest.main()
