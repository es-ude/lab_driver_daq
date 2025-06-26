# Driver for DAQ Devices of the IES Lab
Python Driver for Measurement Devices from the Lab

## Using the package release in other Repos
Just adding this repo in the pyproject.toml file as dependencies by listing the git path.

## Installation
### Python for Developing
We recommended to install all python packages for using this API with a virtual environment (venv). Therefore, we also recommend to `uv` ([Link](https://docs.astral.sh/uv/)) package manager. `uv` is not a standard package installed on your OS. For this, you have to install it in your Terminal (Powershell on Windows) with the following line.
````
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
````
Afterwards with ``uv sync``, the venv will be created and all necessary packages will be installed.
````
uv venv
.\.venv\Scripts\activate  
uv sync
````
### Necessary drivers
For using the modules, it is necessary to install the VISA drivers for your OS in order to identify the USB devices, which will be known as IVI device. If you do not do this, then no access is available.
You can download it [here](https://www.rohde-schwarz.com/de/driver-pages/fernsteuerung/3-visa-und-tools_231388.html).

## How-To-Use
### MXO44 how-to on Linux:
1. Start RsVisa Tester
2. Click on Find Resource once after connecting via USB
3. Done

### DMM6500 how-to on Linux:
You need to do a first time setup
1. Check the USB connection `lsusb`
2. This file may be owned by root, so if that's the case:
`sudo chown <user> /sys/bus/usb/drivers/usbtmc/new_id`
3. Ensure usbtmc kernel module is loaded `sudo modprobe usbtmc`, then `ls /dev/usbtmc*`
and check if the device is listed
4. Create a udev rule to allow access to the device
`echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="05e6", ATTR{idProduct}=="6500", MODE="0666"' | sudo tee /etc/udev/rules.d/99-keithley.rules`
5. Reload the udev rules with `sudo udevadm control --reload-rules` and then `sudo udevadm trigger`
6. Restart the device
7. Crucial: *Open* pyvisa ResourceManager with @py backend instead of IVI-VISA, but let IVI
*scan* for devices beforehand!

### Raspberry Pi Setup
1. Go to https://www.rohde-schwarz.com/ch/applikationen/r-s-visa-application-note_56280-148812.html and install the VISA setup for Raspbian
2. Create then start a virtual environment in the project directory

   `python -m venv venv`

   `source venv/bin/activate`
3. Ensure `setup.py` exists in the project root directory and install via pip `pip install -e .`
4. Run the examples to ensure everything works fine, install missing packages manually with pip if not

## Notes / Errors for using the lab devices
### NGU411 FastLog
FastLog dumps a binary file of its measurements on the USB stick. This binary
file is entirely composed of single precision (32-bit) floating point numbers
in a pair format, that is, the first value is the voltage and the second value
is the current of the measurement. This pair format of measurements repeats
for the entire file. SI-Units are used (V and A). Values are stored in little
endian format. Conversion to CSV should be vastly faster using a simple C
program instead of the machine's built-in converter.

### MXO44
The device freezes upon the detection of a USB (dis-)connection event and will
cease all functionality on firmware version 2.4.2.1. The only fix is cutting
power to the device by holding the power button or plugging it from the socket.

## Build your Script for automated Testing and Plotting
Here is an example for building the test routine of a quantization front-end (it includes a multiplexer, pre-amplifier and analog-digital converter).  

````
from os.path import join, dirname, basename
from glob import glob
from time import sleep
from logging import getLogger, Logger
from lab_driver import scan_instruments, DriverNGUX01
from lab_driver.charac_adc import CharacterizationADC

from src import get_repo_name, get_path_to_project
from src.structure_builder import init_project_folder


class TestHandlerADC:
    _hndl_test: CharacterizationADC
    _hndl_dut: DriverDUT
    _hndl_daq: DriverNGUX01
    __ref_folder: str = get_repo_name()
    _logger: Logger = getLogger(__name__)
    _en_debug: bool = False
    _file_name: str
    _folder_name: str
    _search_index: str='adc'

    def __init__(self, com_dut: str, com_ngu: str, en_debug: bool=False, only_plot: bool=False) -> None:
        """Class for handling the Analog-Digital-Converter test routine
        :param com_dut:     String with COM-Port of DUT board (default: '')
        :param com_ngu:     String with COM-Port of NGU411 DAQ (default: '')
        :param en_debug:    Boolean for enabling debugging mode (without NGU411) (default=False)
        :param only_plot:   Boolean for plotting mode (default=False)
        """
        init_project_folder()
        self._hndl_test = CharacterizationADC(folder_reference=self.__ref_folder)
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
            self._hndl_daq = DriverNGUX01()
            self._hndl_daq.serial_open_known_target(
                resource_name=com_ngu,
                do_reset=True
            )
            self._hndl_daq.do_beep()

    def get_overview_folder(self) -> list:
        """Function to get an overview of available numpyz files"""
        return glob(join(self._folder_name, "*.npz"))

    def run_transfer_test(self) -> dict:
        """Function for running the ADC test on DUT device
        :return:            Dictionary with ['stim': DAC input stream, 'settings': Settings, 'ch<X>': DAQ results with 'val' and 'std']
        """
        self._hndl_daq.output_activate()
        sleep(0.5)
        results = self._hndl_test.run_test_transfer(
            func_mux=self._hndl_dut.change_adc_mux,
            func_daq=self._hndl_daq.set_voltage if not self._en_debug else self._hndl_test.dummy_set_daq,
            func_sens=self._hndl_daq.get_measurement_voltage if not self._en_debug else self._hndl_test.dummy_get_daq,
            func_dut=self._hndl_dut.get_adc_rawdata,
            func_beep=self._hndl_daq.do_beep if not self._en_debug else self._hndl_test.dummy_beep,
        )
        self._hndl_test.save_results(
            file_name=self._file_name,
            data=results,
            settings=self._hndl_test.settings,
            folder_name=self._folder_name,
        )
        self._hndl_daq.output_deactivated()
        return results

    def plot_results_from_measurement(self, results: dict) -> None:
        """Function for plotting the ADC test results
        :param results:     Dictionary from measurement
        :return:            None
        """
        self._hndl_test.plot_characteristic_results_direct(
            data=results,
            file_name=self._file_name,
            path=self._folder_name
        )

    def plot_results_from_file(self, path2file: str) -> None:
        """Function for plotting the ADC test results
        :param path2file:   String with path to file
        :return:            None
        """
        if self._search_index in path2file:
            self._hndl_test.plot_characteristic_results_from_file(
                path=dirname(path2file),
                file_name=basename(path2file)
            )

def run_test() -> None:
    bool_only_plot = False

    print(scan_instruments())
    hndl = TestHandlerADC(
        com_cla='COM7',
        com_ngu='USB0::124:dada:3545',
        en_debug=False,
        only_plot=bool_only_plot
    )

    if not bool_only_plot:
        data = hndl.run_transfer_test()
        hndl.plot_results_from_measurement(data)
    else:
        for file in hndl.get_overview_folder():
            hndl.plot_results_from_file(file)
````
