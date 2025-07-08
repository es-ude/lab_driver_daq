from lab_driver.csv_handler import CsvHandler
from lab_driver.plots import plot_spectral_data, plot_fra_data


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
