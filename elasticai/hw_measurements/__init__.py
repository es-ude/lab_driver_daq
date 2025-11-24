import numpy as np
from dataclasses import dataclass
from .scan_instruments import scan_instruments
from .data_types import (
    TransientData,
    TransformSpectrum,
    FrequencyResponse,
    TransientNoiseSpectrum,
    MetricNoise
)


def get_path_to_project(new_folder: str='', max_levels: int=5) -> str:
    """Function for getting the path to find the project folder structure in application.
    :param new_folder:  New folder path
    :param max_levels:  Max number of levels to get-out for finding pyproject.toml
    :return:            String of absolute path to start the project structure
    """
    from pathlib import Path
    cwd = Path(".").absolute()
    current = cwd

    def is_project_root(p):
        return (p / "pyproject.toml").exists()

    for _ in range(max_levels):
        if is_project_root(current):
            return str(current / new_folder)
        current = current.parent

    if is_project_root(current):
        return str(current / new_folder)
    return str(cwd)


def init_project_folder(new_folder: str='') -> None:
    """Generating folder structure in first run
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    from os import makedirs
    from os.path import join

    folder_structure = ['runs', 'config']
    copy_files = {}

    path2start = join(get_path_to_project(), new_folder)
    makedirs(path2start, exist_ok=True)

    for folder_name in folder_structure:
        makedirs(join(path2start, folder_name), exist_ok=True)


@dataclass(frozen=True)
class DriverPort:
    """Class with COM-Port addresses of each device for testing
    Attributes:
        com_ngu (str):  COM-Port of the R&S NGU411 (Four-Quadrant SMU)
        com_dmm (str):  COM-Port of the Keithley DMM411 (Digital Multimeter)
        com_mxo (str):  COM-Port of the R&S MXO411 (Mixed-Signal Oscilloscope)
        com_hmp (str):  COM-Port of the R&S HMP40x (Power Supply)
    """
    com_ngu: str
    com_dmm: str
    com_mxo: str
    com_hmp: str


DriverPortIES = DriverPort(
    com_ngu='USB0::0x0AAD::0x0197::3639.3763k04-101215::INSTR',
    com_dmm='USB0::0x05E6::0x6500::04622454::INSTR',
    com_mxo='',
    com_hmp='COM7',
)
