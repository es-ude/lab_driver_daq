from os.path import join, dirname, basename
from glob import glob
from time import sleep
from logging import getLogger, Logger
from lab_driver import get_repo_name, get_path_to_project, init_project_folder, DriverNGUX01, DriverDMM6500
from lab_driver.charac.amp import CharacterizationAmplifier
# PYTHON API OF THE DUT HAVE TO BE INCLUDED
from src import DriverDUT


class TestHandlerAmplifier:
    _hndl_test: CharacterizationAmplifier
    _hndl_dut: DriverDUT
    _hndl_src: DriverNGUX01
    _hndl_daq: DriverDMM6500
    __ref_folder: str = get_repo_name()
    _logger: Logger = getLogger(__name__)
    _en_debug: bool = False
    _file_name: str
    _folder_name: str
    _search_index: str='amp'

    def __init__(self, com_dut: str, com_dmm: str, com_ngu: str, en_debug: bool=False, only_plot: bool=False) -> None:
        """Class for handling the Analog-Digital-Converter test routine
        :param com_dut:     String with COM-Port of DUT board
        :param com_dmm:     String with COM-Port of DMM6500 DAQ
        :param com_ngu:     String with COM-Port of NGU411 DAQ
        :param en_debug:    Boolean for enabling debugging mode (without DAQ system) (default=False)
        :param only_plot:   Boolean for plotting mode (default=False)
        """
        init_project_folder()
        self._hndl_test = CharacterizationAmplifier(folder_reference=self.__ref_folder)
        system_id = int(self._hndl_test.settings.system_id)
        self._file_name = f'{self._hndl_test.settings.get_date_string()}_{self._search_index}_charac_id-{system_id:03d}'
        self._folder_name = join(get_path_to_project(), "runs")

        if not only_plot:
            self._hndl_dut = DriverDUT(
                port=com_dut,
                timeout=1.0
            )
        self._en_debug = en_debug
        if not self._en_debug and not only_plot:
            self._hndl_src = DriverNGUX01()
            self._hndl_src.serial_open_known_target(
                resource_name=com_ngu,
                do_reset=True
            )
            self._hndl_src.do_beep()
            sleep(1)

            self._hndl_daq = DriverDMM6500()
            self._hndl_daq.serial_start_known_target(
                resource_name=com_dmm,
                do_reset=True
            )
            self._hndl_daq.do_beep()

    def get_overview_folder(self) -> list:
        """Function to get an overview of available numpyz files"""
        return glob(join(self._folder_name, "*.npz"))

    def run_transfer_test(self, chnnl_num: int) -> dict:
        """Function for running the ADC test on DUT device
        :return:            Dictionary with ['stim': DAC input stream, 'settings': Settings, 'ch<X>': DAQ results with 'val' and 'std']
        """
        self._hndl_src.output_activate()
        sleep(0.5)
        results = self._hndl_test.run_test_transfer(
            chnl=chnnl_num,
            func_mux=self._hndl_dut.change_adc_mux,
            func_set_daq=self._hndl_src.set_voltage if not self._en_debug else self._hndl_test.dummy_set_daq,
            func_sens=self._hndl_src.get_measurement_voltage if not self._en_debug else self._hndl_test.dummy_get_daq,
            func_get_daq=self._hndl_daq.get_voltage,
            func_beep=self._hndl_daq.do_beep if not self._en_debug else self._hndl_test.dummy_beep,
        )
        self._hndl_test.save_results(
            file_name=self._file_name,
            data=results,
            settings=self._hndl_test.settings,
            folder_name=self._folder_name,
        )
        self._hndl_src.output_deactivated()
        return results

    def plot_results_from_measurement(self, results: dict) -> None:
        """Function for plotting the ADC test results
        :param results:     Dictionary from measurement
        :return:            None
        """
        raise NotImplementedError
        """
        self._hndl_test.plot(
            data=results,
            file_name=self._file_name,
            path=self._folder_name
        )
        """

    def plot_results_from_file(self, path2file: str) -> None:
        """Function for plotting the ADC test results
        :param path2file:   String with path to file
        :return:            None
        """
        raise NotImplementedError
        """
        if self._search_index in path2file:
            self._hndl_test.plot_characteristic_results_from_file(
                path=dirname(path2file),
                file_name=basename(path2file)
            )
        """

def run_test() -> None:
    bool_only_plot = False
    hndl = TestHandlerAmplifier(
        com_dut='COM7',
        com_dmm='USB0::124:dada:3545',
        com_ngu='USB0::123:dada:3543',
        en_debug=False,
        only_plot=bool_only_plot
    )

    if not bool_only_plot:
        data = hndl.run_transfer_test()
        hndl.plot_results_from_measurement(data)
    else:
        for file in hndl.get_overview_folder():
            hndl.plot_results_from_file(file)