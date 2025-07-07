import unittest
import numpy as np
from lab_driver import get_path_to_project
from lab_driver.process.data import MetricCalculator, do_fft, calculate_total_harmonics_distortion, calculate_total_harmonics_distortion_from_transient


class TestDataAnalysis(unittest.TestCase):
    path2data = get_path_to_project(new_folder='test_data')
    hndl = MetricCalculator()
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

    def test_metric_mbe_float(self):
        rslt = self.hndl.calculate_error_mbe(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 4.0
        self.assertTrue(chck)

    def test_metric_mbe_numpy(self):
        rslt = self.hndl.calculate_error_mbe(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        chck = type(rslt) == float and rslt == -1.0
        self.assertTrue(chck)

    def test_metric_mae_float(self):
        rslt = self.hndl.calculate_error_mae(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 4.0
        self.assertTrue(chck)

    def test_metric_mae_numpy(self):
        rslt = self.hndl.calculate_error_mae(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        chck = type(rslt) == float and rslt == 1.0
        self.assertTrue(chck)

    def test_metric_mse_float(self):
        rslt = self.hndl.calculate_error_mse(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 16.0
        self.assertTrue(chck)

    def test_metric_mse_numpy(self):
        rslt = self.hndl.calculate_error_mse(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        chck = type(rslt) == float and rslt == 1.0
        self.assertTrue(chck)

    def test_metric_mape_float(self):
        rslt = self.hndl.calculate_error_mape(
            y_pred=2.0,
            y_true=-2.0,
        )
        chck = type(rslt) == float and rslt == 2.0
        self.assertTrue(chck)

    def test_metric_mape_numpy(self):
        rslt = self.hndl.calculate_error_mape(
            y_pred=np.linspace(1.0, 5.0, endpoint=True, num=10),
            y_true=np.linspace(2.0, 6.0, endpoint=True, num=10),
        )
        chck = type(rslt) == float and rslt == 0.28133972977262783
        self.assertTrue(chck)

    def test_metric_thd_one_harmonic(self):
        sampling_rate = 1000
        t = np.linspace(0, 1, sampling_rate, endpoint=True)
        signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.sin(2 * np.pi * 150 * t)

        rslt = self.hndl.calculate_total_harmonics_distortion(
            signal=signal,
            fs=sampling_rate,
            N_harmonics=1
        )
        self.assertEqual(rslt, -20.067970271376048)

    def test_metric_thd_two_harmonic(self):
        sampling_rate = 1000
        t = np.linspace(0, 1, sampling_rate, endpoint=True)
        signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.sin(2 * np.pi * 150 * t)

        rslt = self.hndl.calculate_total_harmonics_distortion(
            signal=signal,
            fs=sampling_rate,
            N_harmonics=2
        )
        self.assertEqual(rslt, -19.118108722018935)

    def test_calculate_cosine_match(self):
        t = np.linspace(0, 1, 1000, endpoint=True)
        rslt = self.hndl.calculate_cosine_similarity(
            y_pred=np.sin(2 * np.pi * 50 * t),
            y_true=np.sin(2 * np.pi * 50 * t),
        )
        self.assertEqual(rslt, 1.0)

    def test_calculate_cosine_half(self):
        t = np.linspace(0, 1, 1000, endpoint=True)
        rslt = self.hndl.calculate_cosine_similarity(
            y_pred=np.sin(2 * np.pi * 50 * t),
            y_true=np.sin(2 * np.pi * 100 * t),
        )
        self.assertEqual(rslt, -4.439780764141639e-17)

    def test_metric_thd_one_harmonic_tran(self):
        sampling_rate = 1000
        t = np.linspace(0, 1, sampling_rate, endpoint=True)
        signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.sin(2 * np.pi * 150 * t)

        rslt = calculate_total_harmonics_distortion_from_transient(
            signal=signal,
            fs=sampling_rate,
            N_harmonics=1
        )
        self.assertEqual(rslt, -20.067970271376048)

    def test_metric_thd_one_harmonic_spec(self):
        sampling_rate = 1000
        t = np.linspace(0, 1, sampling_rate, endpoint=True)
        signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.sin(2 * np.pi * 150 * t)

        freq, spec = do_fft(
            y=signal,
            fs=sampling_rate,
        )

        rslt = calculate_total_harmonics_distortion(
            freq=freq,
            spectral=spec,
            N_harmonics=2
        )
        self.assertEqual(rslt, -19.118108722018935)


if __name__ == "__main__":
    unittest.main()
