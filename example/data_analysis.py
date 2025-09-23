from lab_driver.process.mxo4x import (
    load_spectral_data,
    load_fra_data,
    load_transient_data
)
from lab_driver.plots import (
    plot_spectral_data,
    plot_fra_data,
    plot_transient_data
)
from glob import glob


def analyse_spectral_data(path: str, show_last_plot: bool=False) -> None:
    list_files = glob(f'{path}/*.csv')
    for file_sel in list_files:
        data_spectral = load_spectral_data(
            path=path,
            file_name=file_sel
        )
        plot_spectral_data(
            data=data_spectral,
            file_name=file_sel,
            path2save=path,
            show_plot=show_last_plot and file_sel == list_files[-1]
        )


def analyse_fra_data(path: str, show_last_plot: bool=False) -> None:
    list_files = glob(f'{path}/*.csv')
    for file_sel in list_files:
        data_fra = load_fra_data(
            path=path,
            file_name=file_sel
        )
        plot_fra_data(
            data=data_fra,
            file_name=file_sel,
            path2save=path,
            show_plot=show_last_plot and file_sel == list_files[-1]
        )


def analyse_transient_data(path: str, show_last_plot: bool=True) -> None:
    list_files = glob(f'{path}/*.h5')
    for file_sel in list_files:
        data_tran = load_transient_data(
            path=path,
            file_name=file_sel,
            freq_ref=500.
        )
        plot_transient_data(
            data=data_tran,
            file_name=file_sel,
            path2save=path,
            show_plot=show_last_plot and file_sel == list_files[-1],
            xzoom=[10000, 16000]
        )


if __name__ == "__main__":
    #analyse_fra_data('C:/Users/Andre/Desktop/EEG-CH0')
    analyse_transient_data('C:\\Users\\Andre\\Desktop\\MXO44\\Waveforms')
