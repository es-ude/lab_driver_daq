from pathlib import Path
from elasticai.hw_measurements import TransientData, TransformSpectrum, TransientNoiseSpectrum, MetricNoise
from elasticai.hw_measurements.charac import CharacterizationNoise
from elasticai.hw_measurements.plots import plot_spectrum_noise, plot_transient_noise, plot_spectrum_harmonic
from elasticai.hw_measurements.process import do_fft


def extract_noise_metrics(data: TransientData, exclude_channels: list, path2file: Path,
                          scale_adc: float=1.0, show_plots: bool=True) -> None:
    """Function for extracting the noise spectrum density and the output effect noise voltage from transient noise measurement from single file
    :param data:                Dataclass TransientData with results
    :param exclude_channels:    List with integers to exclude channels from analysis
    :param path2file:           Path to the processed file
    :param scale_adc:           Floating value for scaling the digital values to get a voltage value
    :param show_plots:          If True, show all plots
    :return:                    None
    """
    # --- Processing
    dut = CharacterizationNoise()
    dut.load_data(
        time=data.timestamps,
        signal=data.rawdata
    )
    metrics: MetricNoise = dut.extract_transient_metrics()
    dut.extract_noise_power_distribution(
        scale=scale_adc,
        num_segments=10000
    )
    dut.exclude_channels_from_spec(exclude_channels)
    noise: TransientNoiseSpectrum = dut.remove_power_line_noise(
        tolerance=5.,
        num_harmonics=5
    )

    # --- Plotting
    plot_transient_noise(
        data=[data],
        offset=metrics.offset_mean,
        scale=scale_adc,
        path2save=str(path2file.parent),
        file_name=path2file.stem,
        show_plot=False
    )
    plot_spectrum_noise(
        data=noise,
        path2save=str(path2file.parent),
        file_name=path2file.stem,
        show_plot=False,
    )

    spec: TransformSpectrum = do_fft(
        y=scale_adc * data.rawdata[7, :],
        fs=dut.get_sampling_rate,
        method_window='Hamming',
    )
    plot_spectrum_harmonic(
        data=spec,
        N_harmonics=8,
        path2save=str(path2file.parent),
        file_name=path2file.stem,
        show_plot=show_plots
    )
