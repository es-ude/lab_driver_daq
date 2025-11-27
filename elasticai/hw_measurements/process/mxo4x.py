import h5py
import numpy as np
from pathlib import Path
from elasticai.hw_measurements import TransformSpectrum, FrequencyResponse, TransientData
from elasticai.hw_measurements.csv_handler import CsvHandler
from elasticai.hw_measurements.process.data import do_fft


def load_spectral_data(path: Path, file_name: str, begin_line: int=33) -> TransformSpectrum:
    """Function for loading the csv files of spectrum analysis from R&S MXO44
    :param path:        path to csv file
    :param file_name:   name of csv file
    :param begin_line:  Integer of starting line to extract information from csv
    :return:            Dataclass TransformSpectrum
    """
    data = CsvHandler(
        path=str(path),
        file_name=file_name,
        delimiter=','
    ).read_data_from_csv(
        include_chapter_line=False,
        start_line=begin_line,
        type_load=float
    )
    return TransformSpectrum(
        freq=data[:, 0],
        spec=data[:, 1],
        sampling_rate=1.
    )

def load_fra_data(path2file: Path, begin_line: int=2) -> FrequencyResponse:
    """Function for loading the csv files of Frequency Response Analysis (FRA) from R&S MXO44
    :param path2file:   Path to csv file
    :param begin_line:  Integer of starting line to extract information from csv
    :return:            Dataclass with FrequencyResponse
    """
    data = CsvHandler(
        path=path2file.parent,
        file_name=path2file.stem,
        delimiter=','
    ).read_data_from_csv(
        include_chapter_line=False,
        start_line=begin_line,
        type_load=float
    )
    return FrequencyResponse(
        freq=data[:, 1],
        gain=data[:, 2],
        phase=data[:, 3],
    )


def load_transient_data(path2file: Path) -> TransientData:
    data = list()
    chan = list()
    samp_rate = 0.
    with h5py.File(path2file, 'r') as f:
        key_channels = list(f['Waveforms'].keys())
        # --- Getting sampling rate
        for key, value in f['Waveforms'][key_channels[0]].attrs.items():
            if key == "XInc":
                samp_rate = 1 / float(value)
        # --- Getting data
        for channel in key_channels:
            data.append(f['Waveforms'][channel][f'{channel} Data'][...])
            chan.append(channel)
        f.close()

    data0 = np.array(data)
    if len(data0.shape) == 1:
        data0 = np.expand_dims(data0, axis=0)

    return TransientData(
        rawdata=data0,
        timestamps=np.array([idx/samp_rate for idx in range(len(data[0]))]),
        sampling_rate=samp_rate,
        channels=chan,
    )
