import numpy as np
import pyxdf
from os.path import join
from glob import glob
from lab_driver.charac import CharacterizationNoise
from lab_driver.plots import plot_spectrum_noise, plot_transient_noise


def load_data(path2file: str) -> dict:
    """"""
    data_xdf = pyxdf.load_xdf(
        filename=path2file
    )[0][0]
    num_channels = data_xdf['info']['channel_count'][0]
    time = data_xdf['time_stamps']
    time -= time[0]
    data = np.transpose(data_xdf['time_series'])
    return dict(
        num_channels=num_channels,
        time=time,
        signal=data
    )


if __name__ == '__main__':
    # --- Data Loading
    path2file = "C:/Users/Andre/Desktop/Rauschmessung/ses-2kHz"
    overview_data = glob(join(path2file, "*.xdf"))
    print(overview_data)
    print("\n")
    file_take = overview_data[0]
    data = load_data(file_take)

    # --- Processing
    dut = CharacterizationNoise()
    dut.load_data(
        time=data['time'],
        signal=data['signal']
    )

    metrics = dut.extract_noise_metric()
    dut.extract_noise_power_distribution(
        scale=3.3 / 2 ** 18,
        num_segments=10000
    )
    dut.exclude_channels_from_spec([0, 5])
    noise = dut.remove_power_line_noise(
        tolerance=5.,
        num_harmonics=5
    )
    noise_rms = dut.extract_noise_rms()

    # --- Plotting
    plot_transient_noise(
        time=data["time"],
        signal=data["signal"],
        offset=metrics["offset_mean"],
        path2save=file_take,
        file_name=file_take,
        show_plot=False
    )
    plot_spectrum_noise(
        freq=noise["freq"],
        spec=noise["spec"],
        channels=noise["chan"],
        path2save=file_take,
        file_name=file_take,
        show_plot=True,
    )
