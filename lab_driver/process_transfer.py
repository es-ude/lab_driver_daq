import numpy as np
from logging import getLogger


class ProcessTransferFunction:
    _meas_data: dict

    def __init__(self, data: dict) -> None:
        self._logger = getLogger(__name__)
        self._meas_data = data

    def calculate_inl(self, data: dict) -> np.ndarray:
        return np.zeros(shape=(1,))

    def calculate_dnl(self, data: dict) -> np.ndarray:
        return np.zeros(shape=(1,))
