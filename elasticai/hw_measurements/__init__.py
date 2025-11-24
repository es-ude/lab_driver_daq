from .predefines import DriverPort, DriverPortIES
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
