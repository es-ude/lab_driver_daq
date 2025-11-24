import numpy as np
from logging import getLogger, Logger
from scipy.signal import welch, find_peaks
from elasticai.hw_measurements import TransientNoiseSpectrum, MetricNoise
from elasticai.hw_measurements.plots import scale_auto_value


class CharacterizationNoise:
    _logger: Logger
    _fs: float
    _time: np.ndarray
    _signal: np.ndarray
    _channels: list[str]
    _spec: TransientNoiseSpectrum
    _metric: MetricNoise

    def __init__(self) -> None:
        """Class for analysing transient measurement to extract noise properties"""
        self._logger = getLogger(__name__)

    @property
    def get_sampling_rate(self) -> float:
        """Returning the sampling rate of the measurement"""
        return self._fs

    @property
    def get_channels_overview(self) -> list[int]:
        """Returning a list with available channels to analyse"""
        return self._channels

    @property
    def get_num_channels(self) -> int:
        """Return the number of channels"""
        return len(self._channels)

    def load_data(self, time: np.ndarray, signal: np.ndarray, channels: list) -> None:
        """Function for loading the measurement data into the class
        :param time:    Numpy array with time information [shape: (num of samples, )]
        :param signal:  Numpy array with noise information [shape: (num of channels, num of samples)]
        :return:        None
        """
        if signal.ndim != 2:
            raise ValueError("Signal shape must be (num_channels, data) - Please adapt!")
        if signal.shape[0] > signal.shape[1]:
            raise ValueError("Signal shape must be (num_channels, data) - Please adapt!")

        self._fs = float(1 / np.mean(np.diff(time)))

        self._time = time
        self._signal = signal
        self._channels = channels

    def exclude_channels_from_spec(self, exclude_channel: list) -> None:
        """Function for excluding channels to extract the noise spectrum density
        :param exclude_channel: List of channels to exclude
        :return:                None
        """
        data_freq = self._spec.freq.tolist()
        data_spec = self._spec.spec.tolist()
        data_chan = self._spec.chan

        for idx, item in enumerate(exclude_channel):
            data_freq.pop(item - idx)
            data_spec.pop(item - idx)
            data_chan.pop(item - idx)

        self._spec = TransientNoiseSpectrum(
            freq=np.array(data_freq),
            spec=np.array(data_spec),
            chan=data_chan
        )

    def extract_transient_metrics(self) -> MetricNoise:
        """Function for extracting some metrics from transient measurement data
        :return:    Dataclass MetricNoise with metrics
        """
        if len(self._channels) == 0:
            raise ValueError("Data is not loaded. Please load data first.")

        offset_mean = np.mean(self._signal, axis=-1)
        offset_mead = np.median(self._signal, axis=-1)

        offset = np.tile(offset_mean[:, None], (1, self._signal.shape[1]))
        peak_pos = np.max(self._signal-offset, axis=-1)
        peak_neg = np.min(self._signal-offset, axis=-1)
        peak_peak = peak_pos - peak_neg

        self._metric = MetricNoise(
            offset_mean=offset_mean,
            offset_mead=offset_mead,
            peak_peak=peak_peak,
            sampling_rate=self._fs,
        )
        return self._metric

    def extract_noise_power_distribution(self, scale: float=1.0, num_segments: int=16354) -> TransientNoiseSpectrum:
        """Function to extract noise power distribution from transient measurement
        :param scale:           Floating value to scale the transient measurement, e.g. to scale the digital output to voltage
        :param num_segments:    Number of samples in the noise spectral density
        :return:                Dataclass of TransientNoiseSpectrum
        """
        if len(self._channels) == 0:
            raise ValueError("Data is not loaded. Please load data first.")

        freq = list()
        NPow = list()
        for sig_ch in self._signal:
            offset = np.mean(sig_ch)
            f, Pxx = welch(
                x=scale * (sig_ch - offset),
                window='hann',
                scaling='density',
                fs=self._fs,
                nperseg=2*num_segments,
                return_onesided=True,
            )
            freq.append(f)
            NPow.append(np.sqrt(Pxx))

        self._spec = TransientNoiseSpectrum(
            freq=np.array(freq),
            spec=np.array(NPow),
            chan=self._channels
        )
        return self._spec

    @staticmethod
    def _get_values_around_multiples(data: list, reference: float, num_harmonics: int, tolerance: float=2.0) -> list:
        result = []
        for value in data:
            if abs(value - (1+len(result)) * reference) <= tolerance:
                result.append(value)
            if len(result) == num_harmonics:
                continue
        return result

    def remove_power_line_noise(self, tolerance: float=5., num_harmonics: int=10) -> TransientNoiseSpectrum:
        """Function for removing the power line noise in the spectrum
        :param tolerance:       Floating tolerance value around the power line frequency (= 50 Hz)
        :param num_harmonics:   Number of harmonics to remove
        :return:                Dataclass of TransientNoiseSpectrum
        """
        pl_line_freq = 50.
        if len(self._channels) == 0:
            raise ValueError("Data is not loaded. Please load data first.")

        peak_freq = self._spec.freq[0,:][find_peaks(
            x=self._spec.spec[0,:],
            distance=int(0.9*pl_line_freq / self._spec.freq[0,:][1]),
        )[0]]
        pl_peak_freq = self._get_values_around_multiples(
            data=peak_freq,
            reference=pl_line_freq,
            num_harmonics=num_harmonics,
            tolerance=tolerance
        )
        if pl_peak_freq:
            df = 0.5
            noise_spectrum_new = list()
            for f_ch, noise_ch in zip(self._spec.freq, self._spec.spec):
                for pl_f0 in pl_peak_freq:
                    mask = (f_ch >= pl_f0 - df) & (f_ch <= pl_f0 + df)
                    mask_pos = np.argwhere(mask == True).flatten()
                    if mask_pos.size > 0:
                        noise_ch[mask_pos] = noise_ch[mask_pos[0]-1]
                noise_spectrum_new.append(noise_ch)
        else:
            noise_spectrum_new = self._spec

        return TransientNoiseSpectrum(
            freq=self._spec.freq,
            spec=np.array(noise_spectrum_new),
            chan=self._spec.chan,
        )

    def extract_noise_rms(self) -> np.ndarray:
        """Function for extracting the output effective noise voltage from the spectrum
        :return:        Numpy array with noise RMS of all channels
        """
        if len(self._channels) == 0:
            raise ValueError("Data is not loaded. Please load data first.")

        scale, unit = scale_auto_value(self._spec.spec)
        eff_noise_rms = list()
        for f_ch, noise_ch in zip(self._spec.freq, self._spec.spec):
            noise_eff = np.sqrt(np.trapezoid(
                y=noise_ch ** 2,
                x=f_ch,
            ))
            eff_noise_rms.append(noise_eff)

        eff_noise_rms = np.array(eff_noise_rms)
        print(f"Available RMS noise [{unit}V]: {scale * eff_noise_rms}")
        print(f"Available RMS noise over all channels [{unit}V]: "
              f"{np.mean(scale * eff_noise_rms)} +/- {np.std(scale * eff_noise_rms)} "
              f"(num_channels={self.get_num_channels})")
        return eff_noise_rms
