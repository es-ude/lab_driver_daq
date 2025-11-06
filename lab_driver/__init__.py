from .scan_instruments import scan_instruments
from dataclasses import dataclass


def get_repo_name() -> str:
    """Getting string with repo name"""
    from os.path import dirname
    from pathlib import Path
    import lab_driver

    path_to_import = dirname(lab_driver.__file__)
    return Path(path_to_import).parts[-2]


def get_path_to_project(new_folder: str='', folder_ref: str='') -> str:
    """Function for getting the path to find the project folder structure.
    :param new_folder:      New folder path (optional)
    :param folder_ref:      String with folder reference to start
    :return:                String of absolute path to start the project structure
    """
    from os import getcwd
    from os.path import dirname, join, abspath
    from pathlib import Path

    if get_repo_name() in getcwd() and not folder_ref:
        import lab_driver as ref
        path_to_import = dirname(ref.__file__)
        path_split = Path(path_to_import).parts[:-1]
        path_to_proj = join(*[path_seg for path_seg in path_split], new_folder)
    else:
        path_to_import = join(getcwd().split(folder_ref)[0], folder_ref) if folder_ref else getcwd()
        path_to_proj = join(path_to_import, new_folder)
    return abspath(path_to_proj)


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
