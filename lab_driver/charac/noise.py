import numpy as np
from logging import getLogger, Logger
from scipy.signal import welch, find_peaks
from lab_driver.plots import scale_auto_value


class CharacterizationNoise:
    _logger: Logger
    _fs: float
    _channels: list[int] = []
    _time: np.ndarray
    _signal: np.ndarray
    _freq: list[np.ndarray]
    _spec: list[np.ndarray]
    _metric: dict = dict()

    def __init__(self) -> None:
        """"""
        self._logger = getLogger(__name__)

    @property
    def get_sampling_rate(self):
        return self._fs

    @property
    def get_channels_overview(self):
        return self._channels

    @property
    def get_num_channels(self):
        return len(self._channels)

    def load_data(self, time: np.ndarray, signal: np.ndarray) -> None:
        """"""
        if signal.ndim != 2:
            raise ValueError("Signal shape must be (num_channels, data) - Please adapt!")
        else:
            if signal.shape[0] > signal.shape[1]:
                raise ValueError("Signal shape must be (num_channels, data) - Please adapt!")

        self._fs = float(1 / np.mean(np.diff(time)))
        self._channels = np.arange(signal.shape[0]).tolist()
        self._time = time
        self._signal = signal

    def exclude_channels_from_spec(self, exclude_channel: list) -> None:
        """"""
        for idx, item in enumerate(exclude_channel):
            self._spec.pop(item - idx)
            self._freq.pop(item - idx)
            self._channels.pop(item - idx)

    def extract_noise_metric(self) -> dict:
        """"""
        if len(self._channels) == 0:
            raise ValueError("Data is not loaded. Please load data first.")

        offset_mean = np.mean(self._signal, axis=-1)
        offset_mead = np.median(self._signal, axis=-1)

        offset = np.tile(offset_mean[:, None], (1, self._signal.shape[1]))
        peak_pos = np.max(self._signal-offset, axis=-1)
        peak_neg = np.min(self._signal-offset, axis=-1)
        peak_peak = peak_pos - peak_neg
        self._metric = dict(
            offset_mean=offset_mean,
            offset_mead=offset_mead,
            peak_peak=peak_peak,
            sampling_rate=self._fs,
        )
        return self._metric

    def extract_noise_power_distribution(self, scale: float=1.0, num_segments: int=16354) -> dict:
        """"""
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

        self._spec = NPow
        self._freq = freq
        return dict(
            freq=freq,
            spec=NPow,
            chan=self._channels
        )

    @staticmethod
    def _get_values_around_multiples(data: list, reference: float, num_harmonics: int, tolerance: float=2.0) -> list:
        result = []
        for value in data:
            if abs(value - (1+len(result)) * reference) <= tolerance:
                result.append(value)
            if len(result) == num_harmonics:
                continue
        return result

    def remove_power_line_noise(self, tolerance: float=5., num_harmonics: int=10) -> dict:
        """"""
        pl_line_freq = 50.
        if len(self._channels) == 0:
            raise ValueError("Data is not loaded. Please load data first.")

        peak_freq = self._freq[0][find_peaks(
            x=self._spec[0],
            distance=int(0.9*pl_line_freq / self._freq[0][1]),
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
            for f_ch, noise_ch in zip(self._freq, self._spec):
                for pl_f0 in pl_peak_freq:
                    mask = (f_ch >= pl_f0 - df) & (f_ch <= pl_f0 + df)
                    mask_pos = np.argwhere(mask == True).flatten()
                    if mask_pos.size > 0:
                        noise_ch[mask_pos] = noise_ch[mask_pos[0]-1]
                    noise_spectrum_new.append(noise_ch)
        else:
            noise_spectrum_new = self._spec

        return dict(
            freq=self._freq,
            spec=noise_spectrum_new,
            chan=self._channels,
        )

    def extract_noise_rms(self) -> np.ndarray:
        """"""
        if len(self._channels) == 0:
            raise ValueError("Data is not loaded. Please load data first.")

        scale, unit = scale_auto_value(self._spec[0])
        eff_noise_rms = list()
        for f_ch, noise_ch in zip(self._freq, self._spec):
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
