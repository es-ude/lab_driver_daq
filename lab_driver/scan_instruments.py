from logging import getLogger
from platform import system
from serial.tools import list_ports
import pyvisa


logger = getLogger(__name__)

def __eq_or_none(a, b):
    return a is None or b is None or a == b

def __log_device_name(name):
    logger.debug(f"{name}")

def get_device_vid(device):
    return list(filter(lambda x: x.device == device, list_ports.comports()))[0].vid

def get_device_pid(device):
    return list(filter(lambda x: x.device == device, list_ports.comports()))[0].pid

def scan_instruments(vid = None, pid = None) -> list:
    """Scanning the VISA bus for instruments
    :return:    List of all detected instruments
    """
    
    # new way using serial.tools.list_ports
    if vid is not None or pid is not None:
        devices = [port.device for port in list_ports.comports()
                   if __eq_or_none(port.vid, vid) and __eq_or_none(port.pid, pid)]
        logger.debug("Available devices")
        logger.debug("--------------------------------------")
        map(__log_device_name, devices)
        assert devices != [], "No instruments found!"
        return devices
    
    # legacy code for when no VID and PID are provided
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
    rm.close()
    assert out_dev_adr != [], "No instruments found!"
    return out_dev_adr
