from logging import getLogger
from platform import system
import pyvisa


logger = getLogger(__name__)

def scan_instruments(driver=None) -> list:
    """Scanning the VISA bus for instruments
    :param driver:    Driver implementing scan_com_name()
    :return:    List of all detected instruments
    """
    
    if driver is None:
        if system() == "Linux":
            rm = pyvisa.ResourceManager("/usr/lib/librsvisa.so@ivi")
        else:
            rm = pyvisa.ResourceManager()
        logger.debug(f"\nUsing VISA driver: {rm}")
        obj_inst = rm.list_resources()
        rm.close()
    else:
        obj_inst = driver.scan_com_name()

    logger.debug("Available devices")
    logger.debug("--------------------------------------")
    for idx, inst_name in enumerate(obj_inst):
        logger.debug(f"{idx}: {inst_name}")
        
    assert obj_inst != [], "No instruments found!"
    return obj_inst
