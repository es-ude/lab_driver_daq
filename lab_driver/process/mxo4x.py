import h5py
import numpy as np
from pathlib import Path
from lab_driver.csv_handler import CsvHandler
from lab_driver.process.data import do_fft


def load_spectral_data(path: str, file_name: str, begin_line: int=33) -> dict:
    """Function for loading the csv files of spectrum analysis from R&S MXO44
    :param path:        path to csv file
    :param file_name:   name of csv file
    :param begin_line:  Integer of starting line to extract information from csv
    :return:            Dictionary with results [keys: 'f': frequency, 'Y': spectral info]
    """
    data = CsvHandler(
        path=path,
        file_name=file_name,
        delimiter=','
    ).read_data_from_csv(
        include_chapter_line=False,
        start_line=begin_line,
        type_load=float
    )
    return {'f': data[:, 0], 'Y': data[:, 1]}


def load_fra_data(path: str, file_name: str, begin_line: int=2) -> dict:
    """Function for loading the csv files of Frequency Response Analysis (FRA) from R&S MXO44
    :param path:        path to csv file
    :param file_name:   name of csv file
    :param begin_line:  Integer of starting line to extract information from csv
    :return:            Dictionary with results [keys: 'f': frequency, 'gain': amplitude, 'phase': phase information]
    """
    data = CsvHandler(
        path=path,
        file_name=file_name,
        delimiter=','
    ).read_data_from_csv(
        include_chapter_line=False,
        start_line=begin_line,
        type_load=float
    )
    return {'f': data[:, 1], 'gain': data[:, 2], 'phase': data[:, 3]}


def load_transient_data(path: str, file_name: str, freq_ref: float) -> dict:
    data = dict()
    with h5py.File(Path(path) / file_name, 'r') as f:
        groups = list(f['Waveforms'].keys())
        groups_meta = list(f['Waveforms'][groups[0]].keys())
        for channel in groups:
            data.update({channel: f['Waveforms'][channel][f'{channel} Data'][...]})

    f, Y = do_fft(data[list(data.keys())[0]], 1., 'Hamming')
    samp_rate = freq_ref / f[np.argmax(Y)] * f[-1]
    data.update({"fs": samp_rate})
    return data
