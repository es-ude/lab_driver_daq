from dataclasses import dataclass


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