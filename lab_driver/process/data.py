import numpy as np
from logging import getLogger, Logger
from scipy.signal import find_peaks
from scipy.signal.windows import gaussian
from lab_driver.process.common import ProcessCommon


def window_method(window_size: int, method: str="hamming") -> np.ndarray:
    """Generating window for smoothing transformation method.
    :param window_size:     Integer number with size of the window
    :param method:          Selection of window method ['': None, 'Hamming', 'guassian', 'bartlett', 'blackman']
    :return:                Numpy array with window
    """
    methods_avai = {
        "hamming": np.hamming(window_size),
        "guassian": gaussian(window_size, int(0.16 * window_size), sym=True),
        "hanning": np.hanning(window_size),
        "bartlett": np.bartlett(window_size),
        "blackman": np.blackman(window_size),
    }

    window = np.ones(window_size)
    for key in methods_avai.keys():
        if method.lower() == key:
            window = methods_avai[key]
    return window


def do_fft(y: np.ndarray, fs: float, method_window: str='') -> [np.ndarray, np.ndarray]:
    """Performing the Discrete Fast Fourier Transformation.
    :param y:   Transient input signal
    :param fs:  Sampling rate [Hz]
    :param method_window:   Selected window ['': None, 'Hamming', 'guassian', 'bartlett', 'blackman']
    :return:    Tuple with (1) freq - Frequency and (2) Y - Discrete output
    """
    fft_in = y
    if method_window:
        window = window_method(window_size=y.size, method=method_window)
        fft_in = window * y
    N = y.size // 2
    fft_out = 2 / N * np.abs(np.fft.fft(fft_in))
    fft_out[0] = fft_out[0] / 2
    freq = fs * np.fft.fftfreq(fft_out.size)

    # Taking positive range
    xsel = np.where(freq >= 0)
    fft_out = fft_out[xsel]
    freq = freq[xsel]
    return freq, fft_out


def calculate_total_harmonics_distortion(freq: np.ndarray, spectral: np.ndarray, N_harmonics: int=4) -> float:
    """Calculating the Total Harmonics Distortion (THD) of spectral input
    Args:
        freq:           Array with frequency values for spectral analysis
        spectral:       Array with Spectral input
        N_harmonics:    Number of used harmonics for calculating THD
    Return:
          THD value (in dB) and corresponding frequency positions of peaks
    """
    fsine = freq[np.argmax(spectral).flatten()[0]]
    # --- Limiting the search space
    pos_x0 = np.argwhere(freq >= 0.5 * fsine).flatten()[0]
    pos_x1 = np.argwhere(freq >= (N_harmonics + 1.5) * fsine).flatten()[0]
    search_y = spectral[pos_x0:pos_x1]

    # --- Getting peaks values
    df = np.mean(np.diff(freq))
    xpos, _ = find_peaks(search_y, distance=int(0.8 * fsine / df))
    peaks_y = search_y[xpos]

    # --- Return THD
    return float(20 * np.log10(np.sqrt(np.sum(np.power(peaks_y[1:], 2))) / peaks_y[0]))


def calculate_total_harmonics_distortion_from_transient(signal: np.ndarray, fs: float, N_harmonics: int=4) -> float:
    """Calculating the Total Harmonics Distortion (THD) from transient input
    Args:
        signal:         Array with frequency values for spectral analysis
        fs:             Sampling rate [Hz]
        N_harmonics:    Number of used harmonics for calculating THD
    Return:
          THD value (in dB)
    """
    freq, spectral = do_fft(
        y=signal,
        fs=fs
    )
    return calculate_total_harmonics_distortion(
        freq=freq,
        spectral=spectral,
        N_harmonics=N_harmonics
    )


class MetricCalculator(ProcessCommon):
    _logger: Logger = getLogger(__name__)

    def __init__(self) -> None:
        """Class with constructors for processing the measurement results for extracting device-specific metrics"""
        super().__init__()

    def calculate_lsb_mean(self, stim_input: np.ndarray, daq_output: np.ndarray) -> float:
        """Function for calculating the mean Least Significant Bit (LSB)
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output
        :return:        Float with LSB
        """
        return float(np.mean(self.calculate_lsb(
            stim_input=stim_input,
            daq_output=daq_output
        ), axis=0))

    @staticmethod
    def calculate_gain_from_transfer(stim_input: np.ndarray, src_output: np.ndarray) -> np.ndarray:
        """Function for extracting the gain of an electrical circuit using the transfer function test
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output (should have same unit like stim_input)
        :return:            Numpy array with gain value
        """
        assert src_output.shape == stim_input.shape, "Dimension / shape mismatch"
        return np.diff(src_output) / np.diff(stim_input)


    @staticmethod
    def calculate_lsb(stim_input: np.ndarray, daq_output: np.ndarray) -> np.array:
        """Function for calculating the mean Least Significant Bit (LSB)
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output
        :return:        Float with LSB
        """
        assert daq_output.shape == stim_input.shape, "Dimension / shape mismatch"
        return np.diff(daq_output) / np.diff(stim_input)

    def calculate_dnl(self, stim_input: np.ndarray, daq_output: np.ndarray) -> np.ndarray:
        """Calculating the Differential Non-Linearity (DNL) of a transfer function from DAC/ADC
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output
        :return:            Numpy array with DNL
        """
        return self.calculate_lsb(stim_input, daq_output) / self.calculate_lsb_mean(stim_input, daq_output) - 1

    def calculate_inl(self, stim_input: np.ndarray, daq_output: np.ndarray) -> np.ndarray:
        """Calculating the Integral Non-Linearity (INL) of a transfer function from DAC/ADC
        :param stim_input:  Numpy array with stimulus input
        :param daq_output:  Numpy array with DAQ output
        :return:        Numpy array with INL
        """
        return daq_output - self.calculate_lsb_mean(stim_input, daq_output) * stim_input

    @staticmethod
    def calculate_error_mbe(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
        """Calculating the distance-based metric with mean bias error
        :parm y_pred:       Numpy array or float value from prediction
        :param y_true:      Numpy array or float value from true label
        :return:            Float value with error
        """
        if isinstance(y_true, np.ndarray):
            assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
            return float(np.sum(y_pred - y_true) / y_pred.size)
        else:
            return y_pred - y_true

    @staticmethod
    def calculate_error_mae(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
        """Calculating the distance-based metric with mean absolute error
        :param y_pred:      Numpy array or float value from prediction
        :param y_true:      Numpy array or float value from true label
        :return:            Float value with error
        """
        if isinstance(y_true, np.ndarray):
            assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
            return float(np.sum(np.abs(y_pred - y_true)) / y_pred.size)
        else:
            return float(np.abs(y_pred - y_true))

    @staticmethod
    def calculate_error_mse(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
        """Calculating the distance-based metric with mean squared error
        :param y_pred:      Numpy array or float value from prediction
        :param y_true:      Numpy array or float value from true label
        :return:            Float value with error
        """
        if isinstance(y_true, np.ndarray):
            assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
            return float(np.sum((y_pred - y_true) ** 2) / y_pred.size)
        else:
            return float(y_pred - y_true) ** 2

    @staticmethod
    def calculate_error_mape(y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float:
        """Calculating the distance-based metric with mean absolute percentage error
        :param y_pred:  Numpy array or float value from prediction
        :param y_true:  Numpy array or float value from true label
        :return:        Float value with error
        """
        if isinstance(y_true, np.ndarray):
            assert y_pred.shape == y_true.shape, "Dimension / shape mismatch"
            return float(np.sum(np.abs(y_true - y_pred) / np.abs(y_true)) / y_true.size)
        else:
            return float(abs(y_true - y_pred) / abs(y_true))

    @staticmethod
    def calculate_total_harmonics_distortion(signal: np.ndarray, fs: float, N_harmonics: int) -> float:
        """Calculating the Total Harmonics Distortion (THD) of transient input
        :param signal:      Numpy array with transient signal to extract spectral analysis
        :param fs:          Applied sampling rate [Hz]
        :param N_harmonics: Number of used harmonics for calculating THD
        :return:            THD value (in dB)
        """
        # --- calculate spectral input
        freq, spectral = do_fft(
            y=signal,
            fs=fs,
            method_window='hamming'
        )

        # --- Limiting the search space
        fsine = float(freq[np.argmax(spectral).flatten()][0])
        pos_x0 = np.argwhere(freq >= 0.5 * fsine).flatten()[0]
        pos_x1 = np.argwhere(freq >= (N_harmonics + 1.5) * fsine).flatten()[0]
        search_y = spectral[pos_x0:pos_x1]

        # --- Getting peaks values
        df = np.mean(np.diff(freq))
        xpos, _ = find_peaks(search_y, distance=int(0.8 * fsine / df))
        peaks_y = search_y[xpos]

        return 20 * np.log10(np.sqrt(np.sum(np.power(peaks_y[1:], 2))) / peaks_y[0])

    @staticmethod
    def calculate_cosine_similarity(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculating the Cosine Similarity of two different inputs (same size)
        :param y_pred:      Numpy array or float value from prediction
        :param y_true:      Numpy array or float value from true label
        :return:            Float value with error
        """
        assert y_pred.shape == y_true.shape, "Shape of input is not identical"
        return float(np.dot(y_pred, y_true) / (np.linalg.norm(y_pred) * np.linalg.norm(y_true)))
