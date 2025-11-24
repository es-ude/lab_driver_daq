# pyxdf is just an example, replace with necessary package
import pyxdf
import numpy as np
from pathlib import Path
from elasticai.hw_measurements import TransientData
from elasticai.hw_measurements.plots import (
    plot_spectrum_harmonic,
    plot_fra_data,
    plot_transient_data
)
from elasticai.hw_measurements.process.mxo4x import (
    load_spectral_data,
    load_fra_data,
    load_transient_data
)
from elasticai.hw_measurements.template.noise import extract_noise_properties


def analyse_spectral_data(path: Path, show_last_plot: bool=False) -> None:
    list_files = list(path.glob('*.csv'))
    for file_sel in list_files:
        data_spectral = load_spectral_data(
            path=path.parent,
            file_name=file_sel.stem
        )
        plot_spectrum_harmonic(
            data=data_spectral,
            file_name=file_sel.stem,
            path2save=str(path.parent),
            show_plot=show_last_plot and file_sel == list_files[-1]
        )


def analyse_fra_data(path: Path, show_last_plot: bool=False) -> None:
    list_files = list(path.glob('*.csv'))
    for file_sel in list_files:
        fra = load_fra_data(
            path=path,
            file_name=file_sel.stem
        )
        plot_fra_data(
            data=fra,
            file_name=file_sel.stem,
            path2save=str(path.parent),
            show_plot=show_last_plot and file_sel == list_files[-1]
        )


def analyse_transient_data(path: Path, show_last_plot: bool=True) -> None:
    list_files = list(path.glob('*.h5'))
    for file_sel in list_files:
        data_tran = load_transient_data(
            path=path,
            file_name=file_sel.stem,
            freq_ref=500.
        )
        plot_transient_data(
            data=[data_tran],
            file_name=file_sel.stem,
            path2save=str(path),
            show_plot=show_last_plot and file_sel == list_files[-1],
            xzoom=[10000, 16000]
        )

def load_data(path2file: str) -> TransientData:
    data_xdf = pyxdf.load_xdf(
        filename=path2file
    )[0][0]
    num_channels = data_xdf['info']['channel_count'][0]
    time = data_xdf['time_stamps']
    time -= time[0]
    data = np.transpose(data_xdf['time_series'])
    fs = float(1 / np.mean(np.diff(time)))

    return TransientData(
        channels=num_channels,
        timestamps=time,
        rawdata=data,
        sampling_rate=fs
    )


if __name__ == "__main__":
    path2data = Path('../characterization_v1')
    analyse_fra_data(path2data / 'FRA', False)
    analyse_spectral_data(path2data / 'Harmonic', True)
    extract_noise_properties(
        func_data_loading=load_data,
        overview_data=list((path2data / 'Noise/ses-2kHz').glob("*.xdf")),
        exclude_channels=[],
        scale_adc=3.3 / 2 ** 18,
        show_plots=True
    )
