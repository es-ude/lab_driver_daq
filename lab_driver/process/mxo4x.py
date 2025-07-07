from lab_driver.csv_handler import CsvHandler
from lab_driver.plots import plot_spectral_data, plot_fra_data


def load_spectral_data(path: str, file_name: str, start_idx_line: int=33) -> dict:
    """"""
    data = CsvHandler(
        path=path,
        file_name=file_name,
        delimiter=','
    ).read_data_from_csv(
        include_chapter_line=False,
        start_line=start_idx_line,
        type_load=float
    )
    return {'f': data[:, 0], 'Y': data[:, 1]}


def load_fra_data(path: str, file_name: str, start_idx_line: int=2) -> dict:
    """"""
    data = CsvHandler(
        path=path,
        file_name=file_name,
        delimiter=','
    ).read_data_from_csv(
        include_chapter_line=False,
        start_line=start_idx_line,
        type_load=float
    )
    return {'f': data[:, 1], 'gain': data[:, 2], 'phase': data[:, 3]}


if __name__ == "__main__":
    path0 = 'C:/Users/Andre/Desktop/CH8/spectral'
    file00 = '500Hz_2025-07-07_1_165650_POS.csv'
    file01 = '500Hz_2025-07-07_2_165700_NEG.csv'
    file02 = '500Hz_2025-07-07_3_165717_INPUT.csv'
    data_spectral = load_spectral_data(
        path=path0,
        file_name=file00
    )
    plot_spectral_data(data_spectral, show_plot=False)

    path1 = 'C:/Users/Andre/Desktop/CH8/fra'
    file10 = 'Results_2025-07-07_1_165616_POS.csv'
    file11 = 'Results_2025-07-07_0_165422_NEG.csv'
    data_fra = load_fra_data(
        path=path1,
        file_name=file11
    )
    plot_fra_data(data_fra, show_plot=True)
