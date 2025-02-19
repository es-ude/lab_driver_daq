import platform
import pyvisa

def scan_instruments(do_print=True) -> list:
    """Scanning the VISA bus for instruments
    Args:
        do_print: True to print every detected instrument
    Returns:
        List of all detected instruments
    """
    if platform.system() == "Linux":
        rm = pyvisa.ResourceManager("/usr/lib/librsvisa.so@ivi")
    else:
        rm = pyvisa.ResourceManager()
    obj_inst = rm.list_resources()

    out_dev_adr = list()
    for idx, inst_name in enumerate(obj_inst):
        out_dev_adr.append(inst_name)
        # --- Printing the stuff
        if do_print:
            if idx == 0:
                print(f"\nUsing VISA driver: {rm}")
                print("Available devices")
                print("--------------------------------------")
            print(f"{idx}: {inst_name}")
    return out_dev_adr
