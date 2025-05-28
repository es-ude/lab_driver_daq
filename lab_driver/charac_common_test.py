import unittest
import numpy as np
from lab_driver.charac_common import CharacterizationCommon


class TestCommon(unittest.TestCase):
    def test_common_charac_func_random_voltage_float(self):
        hndl = CharacterizationCommon()
        result = hndl.bypass_mock_daq()
        self.assertTrue(type(result) == float)

    def test_common_charac_func_random_voltage_array(self):
        hndl = CharacterizationCommon()
        result = np.array([hndl.bypass_mock_daq() for _ in range(100)])
        check = result.shape == (100, ) and result.min() >= -1.0 and result.max() <= 1.0
        self.assertTrue(check)


if __name__ == '__main__':
    unittest.main()
