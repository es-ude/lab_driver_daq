import numpy as np
import pyxdf
from os.path import join
from glob import glob
from elasticai.hw_measurements.template.noise import extract_noise_properties


def load_data(path2file: str) -> dict:
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
    path2file = "C:/Users/Andre/Desktop/characterization_v1/Noise/ses-2kHz"
    overview_data = glob(join(path2file, "*.xdf"))
    #exclude_channels = [0, 1, 2, 3, 4, 5, 6, 8]
    exclude_channels = []
    scale_adc = 3.3 / 2 ** 18
    show_plots = True

    extract_noise_properties(
        func_data_loading=load_data,
        overview_data=overview_data,
        exclude_channels=exclude_channels,
        scale_adc=scale_adc,
        show_plots=show_plots
    )
