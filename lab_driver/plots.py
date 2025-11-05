import numpy as np
from os import makedirs
from os.path import join
from pathlib import Path
from matplotlib import pyplot as plt

from lab_driver.process.data import calculate_total_harmonics_distortion, do_fft


def get_plot_color(idx: int) -> str:
    """Getting the color string"""
    sel_color = ['k', 'r', 'b', 'g', 'y', 'c', 'm', 'gray']
    return sel_color[idx % len(sel_color)]


def get_font_size() -> int:
    """Getting the font size for paper work"""
    return 14


def get_plot_marker(idx: int) -> str:
    """Getting the marker for plotting"""
    sel_marker = '.+x_'
    return sel_marker[idx % len(sel_marker)]


def save_figure(fig, path: str | Path, name: str, formats: list=('pdf', 'svg')) -> None:
    """Saving figure in given format
    Args:
        fig:        Matplot which will be saved
        path:       Path for saving the figure
        name:       Name of the plot
        formats:    List with data formats for saving the figures
    Returns:
        None
    """
    makedirs(path, exist_ok=True)
    path2fig = join(path, name)
    for idx, form in enumerate(formats):
        fig.savefig(f"{path2fig}.{form}", format=form)


def scale_auto_value(data: np.ndarray | float) -> [float, str]:
    """Getting the scaling value and corresponding string notation for unit scaling in plots
    Args:
        data:   Array or value for calculating the SI scaling value
    Returns:
        Tuple with [0] = scaling value and [1] = SI pre-unit
    """
    ref_dict = {'T': -4, 'G': -3, 'M': -2, 'k': -1, '': 0, 'm': 1, 'µ': 2, 'n': 3, 'p': 4, 'f': 5}

    value = np.max(np.abs(data)) if isinstance(data, np.ndarray) else data
    str_value = str(value).split('.')
    digit = 0
    if 'e' not in str_value[1]:
        if not str_value[0] == '0':
            # --- Bigger Representation
            sys = -np.floor(len(str_value[0]) / 3)
        else:
            # --- Smaller Representation
            for digit, val in enumerate(str_value[1], start=1):
                if '0' not in val:
                    break
            sys = np.ceil(digit / 3)
    else:
        val = int(str_value[1].split('e')[-1])
        sys = -np.floor(abs(val) / 3) if np.sign(val) == 1 else np.ceil(abs(val) / 3)

    scale = 10 ** (sys * 3)
    units = [key for key, div in ref_dict.items() if sys == div][0]
    return scale, units


def plot_transfer_function_norm(data: dict, path2save: str='',
                                xlabel: str='Stimulus Input', ylabel: str='Stimulus Output',
                                title: str='Transfer Function', file_name: str='', show_plot: bool=True) -> None:
    """Function for plotting the transfer function
    :param data:        Dictionary with extracted values from measurement data
    :param path2save:   Path for saving the figure
    :param xlabel:      Text Label for x-axis
    :param ylabel:      Text Label for y-axis
    :param title:       Text Label for title
    :param file_name:   File name of the saved figure
    :param show_plot:   Boolean for showing the plot
    :return:            None
    """
    val_input = data['stim']
    xaxis = np.linspace(start=val_input[0], stop=val_input[-1], num=9, endpoint=True)

    val_output = np.array([data[key]['mean'] for key in data.keys() if not key == 'stim'])
    yaxis = np.linspace(start=val_output.min(), stop=val_output.max(), num=9, endpoint=True)
    dy = np.diff(yaxis).max()

    plt.figure()
    for idx, key in enumerate(data.keys()):
        if not key == 'stim':
            plt.step(val_input, data[key]['mean'], where='mid', marker='.', c=get_plot_color(idx), label=key)
            plt.fill_between(val_input, data[key]['mean'] - data[key]['std'], data[key]['mean'] + data[key]['std'],
                             step='mid', alpha=0.3, color='gray')

    plt.xticks(xaxis)
    plt.xlim([val_input[0], val_input[-1]])
    plt.yticks(yaxis)
    plt.ylim([val_output.min()-dy, val_output.max()+dy])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    if path2save and file_name:
        save_figure(plt, path2save, f'{file_name.lower()}')
    if show_plot:
        plt.show(block=True)


def plot_transfer_function_metric(data: dict, func: object, path2save: str='',
                                  xlabel: str='Stimulus Input', ylabel: str='Stimulus Output',
                                  title: str='Transfer Function', file_name: str='') -> None:
    """Function for plotting the metric, extracted from the transfer function
    :param data:        Dictionary with pre-processed data from measurement with keys: ['stim', 'ch<x>': {'mean', 'std'}}
    :param func:        Function for calculating the metric
    :param path2save:   Path for saving the figure
    :param xlabel:      Text Label for x-axis
    :param ylabel:      Text Label for y-axis
    :param title:       Text Label for title
    :param file_name:   File name of the saved figure
    :return:            None
    """
    data_metric = {'stim': data['stim']}
    for key in data.keys():
        if not key == 'stim':
            scale_val = 1.0
            metric = func(data['stim'], data[key]['mean'])
            if not metric.size == data['stim'].size:
                metric = np.concatenate((np.array((metric[0], )), metric), axis=0)
            data_metric.update({key: {'mean': metric,
                                   'std': scale_val * data[key]['std']}})

    plot_transfer_function_norm(
        data=data_metric,
        path2save=path2save,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        file_name=file_name
    )


def plot_spectral_data(data: dict, N_harmonics: int=6, file_name: str='', path2save: str='',
                       delta_peaks: int=20, show_peaks: bool=True, show_plot: bool=True, is_input_db: bool=True) -> None:
    """Plotting the spectral data, measured with R&S MXO44
    :param data:        Dictionary with spectral data from measurement
    :param N_harmonics: Number of harmonics for calculation and plot
    :param file_name:   File name of the saved figure
    :param path2save:   Path for saving the figure
    :param delta_peaks: Number of positions around the peaks
    :param show_peaks:  Boolean for highlighting the harmonics
    :param show_plot:   Boolean for showing the plot
    :param is_input_db: Boolean for whether the data is logarithmic [dB]
    :return:            None
    """
    assert [key for key in data.keys()] == ['f', 'Y']

    # --- Plotten
    scalex = 1e-3
    plt.figure()
    plt.loglog(scalex * data['f'], data['Y'], color='k')
    if show_peaks:
        f_zero = data['f'][data['Y'][delta_peaks:].argmax()+delta_peaks]
        xharm = [np.argwhere(data['f'] >= f_zero * (1+ite)).flatten()[0] for ite in range(1+N_harmonics)]
        for idx, xpos in enumerate(xharm):
            xval = np.linspace(start=xpos-delta_peaks, stop=xpos+delta_peaks, endpoint=False, num=2*delta_peaks, dtype=int)
            plt.loglog(scalex * data['f'][xval], data['Y'][xval], color='r' if idx == 0 else 'b')

    plt.xlim([data['f'][0] * scalex, data['f'][-1] * scalex])
    #plt.xticks(ticks=np.round(np.linspace(data['f'][0], data['f'][-1], 9, dtype=float) * scalex, 1))
    plt.xlabel(r'Frequency $f$ [kHz]', fontsize=get_font_size())
    plt.ylabel(r'Spectral Amplitude $\hat{Y}(f)$ [dB]', fontsize=get_font_size())

    thd = calculate_total_harmonics_distortion(
        freq=data['f'][delta_peaks:],
        spectral=data['Y'][delta_peaks:] if not is_input_db else 10 ** (data['Y'][delta_peaks:] / 20),
        N_harmonics=N_harmonics
    )
    plt.title(f'THD = {thd:.2f} dB', fontsize=get_font_size())
    plt.grid()
    plt.tight_layout()

    if path2save:
        filename_wo_ext = Path(file_name).stem
        save_figure(plt, path=path2save, name=f'{filename_wo_ext}_spectral', formats=['pdf', 'svg', 'eps'])
    if show_plot:
        plt.show(block=True)


def plot_fra_data(data: dict, file_name: str='', path2save: str='',
                  show_plot: bool=True) -> None:
    """Plotting the data from Frequency Response Analysis (FRA) using R&S MXO44
    :param data:        Dictionary with measured data from device
    :param file_name:   File name of the saved figure
    :param path2save:   Path for saving the figure
    :param show_plot:   Boolean for showing the plot
    """
    assert [key for key in data.keys()] == ['f', 'gain', 'phase']
    # --- Preprocessing (Unwrap phase information)
    xphase_jmp = [idx+1 for idx, val in enumerate(np.diff(data['phase'])) if val > +250]
    xphase_art = [True for val in np.diff(data['phase']) if val > +250]
    xphase_jmp.extend([idx+1 for idx, val in enumerate(np.diff(data['phase'])) if val < -250])
    xphase_art.extend([False for val in np.diff(data['phase']) if val < -250])
    phase = data['phase']
    for xpos, style in zip(xphase_jmp, xphase_art):
        phase[xpos:] += -360. if style else +360.

    # --- Extract features
    num_pol = 1
    print(f"Gain_max = {data['gain'].max():.2f} dB")
    xcorner = np.argwhere(data['gain'] - (data['gain'].max()- num_pol*3) < 0).flatten()
    if xcorner.size > 0:
        print(f"f_-3dB = {1e-3* data['f'][xcorner[0]]:.4f} kHz")

    # --- Plot
    fig, ax1 = plt.subplots()
    ax1.semilogx(data['f'], data['gain'], color='k', marker='.', markersize=6)

    ax1.set_xlim([data['f'][0], data['f'][-1]])
    ax1.set_xlabel(r'Frequency $f$ [Hz]', fontsize=get_font_size())
    ax1.set_ylabel(r'Gain $|H(f)|$ [dB]', color='k', fontsize=get_font_size())
    ax1.grid(True, which="both", ls="--")
    ax1.yaxis.get_ticklocs(minor=True)
    ax1.minorticks_on()

    ax2 = ax1.twinx()
    ax2.semilogx(data['f'], phase, color='r', marker='.', markersize=6)
    ax2.set_ylabel(r'Phase $\alpha$ [°]', color='r', fontsize=get_font_size())
    plt.tight_layout()

    if path2save and file_name:
        filename_wo_ext = Path(file_name).stem
        save_figure(plt, path=path2save, name=f'{filename_wo_ext}_fra',  formats=['pdf', 'svg', 'eps'])
    if show_plot:
        plt.show(block=True)


def plot_transient_data(data: dict, file_name: str='', path2save: str='', show_plot: bool=False, xzoom: list=[0, -1]) -> None:
    """Plotting content from transient measurements for extracting Total Harmonic Distortion (THD)
    :param data:        Dictionary with measured data from device with keys: [fs, ...]
    :param file_name:   String with file name of the saved figure
    :param path2save:   String with path for saving the figure
    :param show_plot:   Boolean for showing the plot
    :param xzoom:       List with xzoom values
    :return:            None
    """
    for key, data_ch in data.items():
        if not key == "fs":
            time = np.linspace(start=0, stop=data_ch.size, num=data_ch.size) / data["fs"]
            f, Y = do_fft(data_ch, data["fs"], 'Hamming')
            plot_spectral_data(
                data={"f": 2*f, "Y": Y},
                N_harmonics=10,
                file_name=file_name,
                path2save=path2save,
                show_plot=False,
                is_input_db=False
            )
            f_start = np.power(10, np.floor(np.log10(f[np.argmax(Y)])))

            fig, axs = plt.subplots(nrows=2, ncols=1)
            axs[0].plot(time, data_ch, 'k', label=key)
            axs[0].set_xlim([time[xzoom[0]], time[xzoom[1]]])
            axs[1].loglog(2*f, Y, 'k', label=key)
            axs[1].set_xlim([f_start, f[-1]])
            for ax in axs:
                ax.grid()
            plt.tight_layout()

            if path2save and file_name:
                filename_wo_ext = Path(file_name).stem
                save_figure(plt, path=path2save, name=f'{filename_wo_ext}_transient', formats=['pdf', 'svg', 'eps'])
            if show_plot:
                plt.show(block=True)


def plot_transient_noise(time: np.ndarray, signal: np.ndarray, offset: np.ndarray,
                         xzoom: list=[0, -1], file_name: str="noise", path2save: str="", show_plot: bool=False) -> None:
    """Plotting content from transient measurements for extracting noise properties
    :param time:        Numpy array with time vector
    :param signal:      Numpy array with content, shape: (num_channels, data)
    :param offset:      Numpy array with offset, shape: (num_channels, )
    :param xzoom:       List with xzoom values
    :param file_name:   String with file name of the saved figure
    :param path2save:   String with path for saving the figure
    :param show_plot:   Boolean for showing the plot
    :return:            None
    """
    plt.figure()
    for idx, (dat0, off0) in enumerate(zip(signal, offset)):
        plt.plot(time, dat0 - off0, label=f"CH{idx}", color=get_plot_color(idx))

    plt.xlabel("Time / s", size=get_font_size())
    plt.ylabel("ADC output", size=get_font_size())
    plt.xlim([time[xzoom[0]], time[xzoom[1]]])
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    if path2save and file_name:
        save_path = Path(path2save).parent
        filename_wo_ext = Path(file_name).stem
        save_figure(plt, path=save_path, name=f'{filename_wo_ext}_transient', formats=['pdf', 'svg', 'eps'])
    if show_plot:
        plt.show()


def plot_spectrum_noise(freq: list, spec: list, channels: list, file_name: str="noise", path2save: str="", show_plot: bool=False) -> None:
    """Plotting the noise amplitude spectral density from transient measurements for extracting noise properties
    :param freq:        List with numpy array with frequency vector for each channel
    :param spec:        List with numpy array with noise spectral density vector for each channel
    :param channels:    List of channel names
    :param file_name:   String with file name of the saved figure
    :param path2save:   String with path for saving the figure
    :param show_plot:   Boolean for showing the plot
    :return:            None
    """
    scale_y, unit_y = scale_auto_value(spec[0])
    freq_min = np.min(np.array([f[0] if not f[0] == 0. else f[1] for f in freq]))
    freq_max = np.max(np.array([f.max() for f in freq]))
    freq_dec_max = 10 ** np.ceil(np.log10(freq_max))

    plt.figure()
    for idx, (f, Y, ch) in enumerate(zip(freq, spec, channels)):
        plt.loglog(f, scale_y * Y, label=f"CH{ch}", color=get_plot_color(idx))

    plt.legend(loc="upper left")
    plt.xlim([freq_min, freq_dec_max])
    plt.ylabel("Noise spectral density / " + unit_y + r"V/$\sqrt{Hz}$", size=get_font_size())
    plt.xlabel("Frequency / Hz", size=get_font_size())
    plt.tight_layout()
    plt.grid(True)

    if path2save and file_name:
        save_path = Path(path2save).parent
        filename_wo_ext = Path(file_name).stem
        save_figure(plt, path=save_path, name=f'{filename_wo_ext}_spectrum', formats=['pdf', 'svg', 'eps'])
    if show_plot:
        plt.show()
