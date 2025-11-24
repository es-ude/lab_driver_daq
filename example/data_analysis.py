import numpy as np
from pathlib import Path
from elasticai.hw_measurements import TransientData, TransformSpectrum
from elasticai.hw_measurements.plots import (
    plot_spectrum_harmonic,
    plot_fra_data,
    plot_transient_data
)
from elasticai.hw_measurements.process import (
    do_fft,
    load_fra_data,
    load_transient_data
)
from elasticai.hw_measurements.template.noise import extract_noise_metrics


def load_dut_data(path2file: Path) -> TransientData:
    # pyxdf is just an example, replace with necessary package
    import pyxdf
    data_xdf = pyxdf.load_xdf(
        filename=path2file
    )[0][0]
    num_channels = [f"CH{idx}" for idx in range(int(data_xdf['info']['channel_count'][0]))]
    scale_adc = 3.3 / 2 ** 18

    time = data_xdf['time_stamps']
    time -= time[0]
    fs = float(1 / np.mean(np.diff(time)))

    data = np.transpose(data_xdf['time_series'])
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    return TransientData(
        channels=num_channels,
        timestamps=time,
        rawdata=scale_adc * data,
        sampling_rate=fs
    )


def analyse_fra_mxo4(path: Path, subfolder_name: str, show_last_plot: bool=False) -> None:
    list_files = list((path / subfolder_name).glob('*.csv'))
    for file_sel in list_files:
        fra = load_fra_data(
            path2file=file_sel
        )
        plot_fra_data(
            data=fra,
            file_name=file_sel.stem,
            path2save=str(file_sel.parent),
            show_plot=show_last_plot and file_sel == list_files[-1]
        )

def analyse_harmonic_mxo4(path: Path, subfolder_name: str, show_last_plot: bool=False) -> None:
    list_files = list((path / subfolder_name).glob('*.h5'))
    for file_sel in list_files:
        data_transient = load_transient_data(
            path2file=file_sel,
            freq_ref=500.
        )
        plot_transient_data(
            data=data_transient,
            file_name=file_sel.stem,
            path2save=str(file_sel.parent),
            show_plot=show_last_plot and file_sel == list_files[-1]
        )


def analyse_harmonic_dut(path: Path, subfolder_name: str, take_channel: int, show_last_plot: bool=True) -> None:
    list_files = list((path / subfolder_name).glob('*.xdf'))
    for file_sel in list_files:
        data_tran = load_dut_data(file_sel.absolute())
        data_spec: TransformSpectrum = do_fft(
            y=data_tran.rawdata[take_channel, :],
            fs=data_tran.sampling_rate,
            method_window='Hamming',
        )
        plot_spectrum_harmonic(
            data=data_spec,
            N_harmonics=8,
            path2save=str(file_sel.parent),
            file_name=file_sel.stem,
            is_input_db=False,
            show_plot=show_last_plot and file_sel == list_files[-1],
        )


def analyse_noise_dut(path: Path, subfolder_name: str, exclude_channels: list=(), show_last_plot: bool=False) -> None:
    list_files = list((path / subfolder_name).glob("*.xdf"))
    print("Processing noise files:\n=========================")
    for file_take in list_files:
        data: TransientData = load_dut_data(file_take.absolute())
        extract_noise_metrics(
            data=data,
            exclude_channels=exclude_channels,
            path2file=file_take,
            scale_adc=1.0,
            show_plots=show_last_plot and file_take == list_files[-1],
        )


if __name__ == "__main__":
    path2data = Path('data')
    exclude_noise_channels = [7]

    analyse_fra_mxo4(path2data, subfolder_name='FRA_v1', show_last_plot=False)
    analyse_harmonic_mxo4(path2data, subfolder_name='Harmonic_v1', show_last_plot=False)
    analyse_harmonic_dut(path2data, subfolder_name='Noise_v1/ses-2kHz', take_channel=exclude_noise_channels[0], show_last_plot=False)
    analyse_noise_dut(path2data, subfolder_name='Noise_v1/ses-2kHz', exclude_channels=exclude_noise_channels, show_last_plot=True)
