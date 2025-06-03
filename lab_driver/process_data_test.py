import unittest
import numpy as np
from lab_driver import get_path_to_project
from lab_driver.process_data import ProcessTransferFunction


class TestDataAnalysis(unittest.TestCase):
    path2data = get_path_to_project(new_folder='test_data')
    hndl = ProcessTransferFunction()
    ovr = hndl.get_data_overview(
        path=path2data,
        acronym='dac'
    )
    trns = hndl.process_data_from_file(
        path=path2data,
        filename=ovr[0]
    )

    def test_get_data_overview(self):
        self.assertTrue(len(self.ovr) > 0)

    def test_get_data_length(self):
        self.assertTrue(self.trns['stim'].size == self.trns['ch00']['mean'].size == self.trns['ch00']['std'].size)

    def test_get_data_mean_value(self):
        np.testing.assert_array_equal(self.trns['stim'], self.trns['ch00']['mean'])

    def test_get_data_std_value(self):
        np.testing.assert_array_equal(self.trns['ch00']['std'], np.zeros_like(self.trns['ch00']['mean']))

    def test_lsb_constructor_size(self):
        rslt = self.hndl.calculate_lsb(
            stim_input=self.trns['stim'],
            daq_output=self.trns['ch00']['mean']
        )
        self.assertTrue(rslt.size == self.trns['stim'].size - 1)

    def test_lsb_constructor_value(self):
        rslt = self.hndl.calculate_lsb(
            stim_input=self.trns['stim'],
            daq_output=self.trns['ch00']['mean']
        )
        np.testing.assert_array_equal(rslt, np.ones_like(rslt))

    def test_lsb_constructor_float(self):
        rslt = self.hndl.calculate_lsb_mean(
            stim_input=self.trns['stim'],
            daq_output=self.trns['ch00']['mean']
        )
        self.assertEqual(rslt, 1.0)

    def test_dnl_constructor_size(self):
        rslt = self.hndl.calculate_dnl(
            stim_input=self.trns['stim'],
            daq_output=self.trns['ch00']['mean']
        )
        self.assertTrue(rslt.size == self.trns['stim'].size-1)

    def test_dnl_constructor_value(self):
        rslt = self.hndl.calculate_dnl(
            stim_input=self.trns['stim'],
            daq_output=self.trns['ch00']['mean']
        )
        np.testing.assert_array_equal(rslt, np.zeros_like(rslt))


if __name__ == "__main__":
    unittest.main()
