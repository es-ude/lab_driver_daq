import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class TransientData:
    rawdata: np.ndarray
    timestamps: np.ndarray
    channels: list
    sampling_rate: float

    @property
    def size(self) -> int:
        return self.rawdata.shape[-1]

    @property
    def num_channels(self) -> int:
        return self.rawdata.shape[0]


@dataclass(frozen=True)
class TransformSpectrum:
    freq: np.ndarray
    spec: np.ndarray
    sampling_rate: float


@dataclass(frozen=True)
class FrequencyResponse:
    freq: np.ndarray
    gain: np.ndarray
    phase: np.ndarray


@dataclass(frozen=True)
class TransientNoiseSpectrum:
    freq: np.ndarray
    spec: np.ndarray
    chan: list


@dataclass(frozen=True)
class MetricNoise:
    offset_mean: np.ndarray
    offset_mead: np.ndarray
    peak_peak: np.ndarray
    sampling_rate: float