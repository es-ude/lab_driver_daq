from .dmm6500 import DriverDMM6500
from .mxo4x import DriverMXO4X
from .ngux01 import DriverNGUX01
from .hmp40x0 import DriverHMP40X0
from .rtm3004 import DriverRTM3004
from .scan_instruments import scan_instruments


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
