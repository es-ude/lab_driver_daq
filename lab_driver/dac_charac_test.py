import unittest
import numpy as np
from copy import deepcopy
from lab_driver.dac_charac import SettingsDAC, CharacterizationDAC


settings = SettingsDAC(
    dac_reso=16,
    dac_chnl=[idx for idx in range(4)],
    dac_rang=[0, 2**16-1],
    num_rpt=1,
    num_steps=1,
    daq_ovr=4
)


class TestDAC(unittest.TestCase):
    def test_settings_date(self):
        date = settings.get_date_string()
        check = len(date.split('-')[0]) == 8 and len(date.split('-')[1]) == 6
        self.assertTrue(check)

    def test_settings_stimuli_16bit_1step(self):
        set0 = deepcopy(settings)
        set0.dac_reso = 16
        set0.dac_rang = [0, 2 ** set0.dac_reso - 1]
        set0.num_steps = 1
        stimuli = set0.get_cycle_stimuli_input()
        check = np.array([idx for idx in range(set0.dac_rang[0], set0.dac_rang[1]+1, set0.num_steps)], dtype=int)
        np.testing.assert_array_equal(stimuli, check)

    def test_settings_stimuli_8bit_4step(self):
        set0 = deepcopy(settings)
        set0.dac_reso = 8
        set0.dac_rang = [0, 2 ** set0.dac_reso - 1]
        set0.num_steps = 8
        stimuli = set0.get_cycle_stimuli_input()
        check = np.array([idx for idx in range(set0.dac_rang[0], set0.dac_rang[1] + 1, set0.num_steps)], dtype=int)
        np.testing.assert_array_almost_equal(stimuli, check, decimal=-1)

    def test_settings_stimuli_8bit_16step(self):
        set0 = deepcopy(settings)
        set0.dac_reso = 8
        set0.dac_rang = [0, 2**set0.dac_reso-1]
        set0.num_steps = 16
        stimuli = set0.get_cycle_stimuli_input()
        check = np.array([idx for idx in range(set0.dac_rang[0], set0.dac_rang[1]+1, set0.num_steps)], dtype=int)
        np.testing.assert_array_almost_equal(stimuli, check, decimal=-2)

    def test_settings_result_array(self):
        set0 = deepcopy(settings)
        set0.dac_reso = 8
        set0.dac_rang = [0, 2 ** set0.dac_reso - 1]
        set0.num_steps = 16
        set0.num_rpt = 10

        result_shape = set0.get_cycle_empty_array().shape
        check_shape = (set0.num_rpt, int((set0.dac_rang[1] - set0.dac_rang[0] + 1) / set0.num_steps), set0.daq_ovr)
        np.testing.assert_equal(result_shape, check_shape)

    def test_run_transfer_wo_ovr(self):
        set0 = deepcopy(settings)
        set0.dac_reso = 8
        set0.dac_rang = [0, 2 ** set0.dac_reso - 1]
        set0.num_steps = 16
        set0.num_rpt = 2

        hndl = CharacterizationDAC()
        hndl.settings = set0
        hndl.check_settings_error()
        results = hndl.run_test_dac_transfer(
            func_mux=hndl.bypass_mock_mux,
            func_dac=hndl.bypass_mock_dac,
            func_daq=hndl.bypass_mock_daq,
            func_beep=hndl.bypass_mock_beep
        )
        self.assertTrue(len(results) == 1 + len(set0.dac_chnl))


if __name__ == '__main__':
    unittest.main()
