from logging import getLogger
from platform import system
import pyvisa


logger = getLogger(__name__)


def scan_instruments() -> list:
    """Scanning the VISA bus for instruments
    :return:    List of all detected instruments
    """
    if system() == "Linux":
        rm = pyvisa.ResourceManager("/usr/lib/librsvisa.so@ivi")
    else:
        rm = pyvisa.ResourceManager()
    obj_inst = rm.list_resources()

    logger.debug(f"\nUsing VISA driver: {rm}")
    logger.debug("Available devices")
    logger.debug("--------------------------------------")

    out_dev_adr = list()
    for idx, inst_name in enumerate(obj_inst):
        out_dev_adr.append(inst_name)
        logger.debug(f"{idx}: {inst_name}")
    return out_dev_adr
