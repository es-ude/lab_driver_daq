import unittest
import numpy as np
from copy import deepcopy
from elasticai.hw_measurements.charac.amp import SettingsAmplifier, CharacterizationAmplifier


settings = SettingsAmplifier(
    system_id='0',
    vss=-5.0,
    vdd=5.0,
    test_rang=[0.0, 5.0],
    daq_ovr=4,
    num_rpt=1,
    delta_steps=0.05,
    sleep_sec=0.01
)


class TestAmplifier(unittest.TestCase):
    def test_settings_date(self):
        date = settings.get_date_string()
        check = len(date.split('-')[0]) == 8 and len(date.split('-')[1]) == 6
        self.assertTrue(check)

    def test_settings_num_steps(self):
        set0 = deepcopy(settings)
        set0.test_rang = [-5.0, 5.0]

        points_result = []
        points_check = [201, 101, 21, 11, 5]
        delta_list = [0.05, 0.1, 0.5, 1, 2.5]
        for step in delta_list:
            set0.delta_steps = step
            points_result.append(set0.get_num_steps())
        np.testing.assert_array_equal(points_result, points_check)

    def test_settings_vcm(self):
        set0 = deepcopy(settings)
        set0.adc_rang = [-5.0, 5.0]

        self.assertEqual(set0.get_common_mode_voltage(), 0.0)

    def test_settings_stimuli_one_step(self):
        set0 = deepcopy(settings)
        set0.test_rang = [-5.0, 5.0]
        set0.delta_steps = 1.0

        stimuli = set0.get_cycle_stimuli_input()
        check = np.array([-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.])
        np.testing.assert_array_equal(stimuli, check)

    def test_settings_stimuli_half_step(self):
        set0 = deepcopy(settings)
        set0.test_rang = [0.0, 5.0]
        set0.delta_steps = 0.5

        stimuli = set0.get_cycle_stimuli_input()
        check = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
        np.testing.assert_array_almost_equal(stimuli, check, decimal=1)

    def test_settings_result_array(self):
        set0 = deepcopy(settings)
        set0.test_rang = [-5.0, 5.0]
        set0.delta_steps = 0.5
        set0.num_rpt = 10

        result_shape = set0.get_cycle_empty_array().shape
        check_shape = (set0.num_rpt, set0.get_num_steps(), set0.daq_ovr)
        np.testing.assert_equal(result_shape, check_shape)

    def test_run_transfer_wo_ovr(self):
        set0 = deepcopy(settings)
        set0.test_rang = [-5.0, 5.0]
        set0.delta_steps = 0.5
        set0.num_rpt = 2

        hndl = CharacterizationAmplifier()
        hndl.settings = set0
        results = hndl.run_test_transfer(
            chnl=0,
            func_mux=hndl.dummy_set_mux,
            func_set_daq=hndl.dummy_set_daq,
            func_sens=hndl.dummy_get_daq,
            func_get_daq=hndl.dummy_get_daq,
            func_beep=hndl.dummy_beep
        )
        self.assertTrue(len(results) == 1 + set0.num_rpt)

if __name__ == '__main__':
    unittest.main()
