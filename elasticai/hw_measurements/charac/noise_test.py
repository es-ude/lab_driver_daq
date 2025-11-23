import numpy as np
from unittest import TestCase, main
from elasticai.hw_measurements.charac import CharacterizationNoise


class TestNoise(TestCase):
    def setUp(self):
        self.noise_density = 100e-9  # V/√Hz
        fs = 2e4  # Hz
        duration = 0.5  # seconds

        sigma = self.noise_density * np.sqrt(fs / 2)
        time = np.linspace(start=0.0, stop=duration, num=int(fs * duration), endpoint=True)
        signal = np.random.normal(0, sigma, time.size)
        # signal += np.sin(2*np.pi*time*2)

        self.dut = CharacterizationNoise()
        self.dut.load_data(
            time=time,
            signal=np.expand_dims(signal, axis=0)
        )

    def test_load_data_wrong_format_single(self):
        with self.assertRaises(ValueError):
            self.dut.load_data(
                time=np.linspace(start=0.0, stop=1.0, num=101, endpoint=True),
                signal=np.zeros((101, ))
            )

    def test_load_data_wrong_format_dual(self):
        with self.assertRaises(ValueError):
            self.dut.load_data(
                time=np.linspace(start=0.0, stop=1.0, num=101, endpoint=True),
                signal=np.zeros((101, 1))
            )

    def test_sampling_rate(self):
        assert self.dut.get_sampling_rate == 19998.

    def test_num_channels(self):
        assert self.dut.get_num_channels == 1

    def test_noise_spectrum_resistor(self):
        rslt = self.dut.extract_noise_power_distribution(
            scale=1.0,
            num_segments=1000
        )
        np.testing.assert_array_almost_equal(rslt["spec"], np.zeros_like(rslt["spec"]) + self.noise_density, decimal=7)

    def test_noise_rms_resistor(self):
        self.dut.extract_noise_power_distribution(
            scale=1.0,
            num_segments=100
        )
        rslt = self.dut.extract_noise_rms()
        print(rslt[0])

        assert len(rslt) == self.dut.get_num_channels
        assert 9.75e-6 < rslt[0] < 10.25e-6


if __name__ == '__main__':
    main()
