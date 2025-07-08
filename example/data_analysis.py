from os.path import join
from lab_driver import get_path_to_project
from lab_driver.process.mxo4x import load_spectral_data, load_fra_data
from lab_driver.plots import plot_spectral_data, plot_fra_data


if __name__ == "__main__":
    path2save = join(get_path_to_project(), 'runs')
    sel_file = 0
    path0 = 'C:/Users/Andre/Desktop/CH8/spectral'
    file0 = ['500Hz_2025-07-07_1_165650_POS.csv',
             '500Hz_2025-07-07_2_165700_NEG.csv',
             '500Hz_2025-07-07_3_165717_INPUT.csv']
    data_spectral = load_spectral_data(
        path=path0,
        file_name=file0[sel_file]
    )
    plot_spectral_data(data_spectral, file_name=file0[sel_file], path2save=path2save, show_plot=False)

    path1 = 'C:/Users/Andre/Desktop/CH8/fra'
    file1 = ['Results_2025-07-07_1_165616_POS.csv',
             'Results_2025-07-07_0_165422_NEG.csv',
             'Results_2025-07-07_0_165422_NEG.csv']
    data_fra = load_fra_data(
        path=path1,
        file_name=file1[sel_file]
    )
    plot_fra_data(data_fra, file_name=file1[sel_file], path2save=path2save, show_plot=True)

