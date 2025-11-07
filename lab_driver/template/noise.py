from lab_driver.charac import CharacterizationNoise
from lab_driver.plots import plot_spectrum_noise, plot_transient_noise, plot_spectrum_harmonic
from lab_driver.process.data import do_fft


def extract_noise_properties(func_data_loading, overview_data: list, exclude_channels: list,
                             scale_adc: float=1.0, show_plots: bool=True) -> None:
    """Function for extracting the noise spectrum density and the output effect noise voltage from transient noise measurement
    :param func_data_loading:   Function to load the data with input like path to file
    :param overview_data:       List with strings of the file overview in the folder
    :param exclude_channels:    List with integers to exclude channels from analysis
    :param scale_adc:           Floating value for scaling the digital values to get a voltage value
    :param show_plots:          If True, show all plots
    :return:                    None
    """
    print("Processing noise files:\n=========================")
    for file_take in overview_data:
        print(file_take)
        # --- Data Loading
        data = func_data_loading(file_take)

        # --- Processing
        dut = CharacterizationNoise()
        dut.load_data(
            time=data['time'],
            signal=data['signal']
        )

        metrics = dut.extract_transient_metrics()
        dut.extract_noise_power_distribution(
            scale=scale_adc,
            num_segments=10000
        )
        dut.exclude_channels_from_spec(exclude_channels)
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
            scale=scale_adc,
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
            show_plot=False,
        )

        f, Y = do_fft(
            y=scale_adc * data["signal"][7, :],
            fs=dut.get_sampling_rate,
            method_window='Hamming',
        )
        plot_spectrum_harmonic(
            data={"f": f, "Y": Y},
            N_harmonics=8,
            file_name=file_take,
            path2save=file_take,
            show_plot=file_take == overview_data[-1] and show_plots,
        )
