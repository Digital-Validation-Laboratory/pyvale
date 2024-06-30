'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Callable
import numpy as np

from pyvale.uncertainty.errorcalculator import ErrCalculator


class SysErrRoundOff(ErrCalculator):
    def __init__(self, method: str = 'round', base: float = 1.0) -> None:

        self._base = base
        self._method = select_round_method(method)


    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        rounded_measurements = self._base*self._method(err_basis/self._base)
        return rounded_measurements - err_basis


class SysErrDigitisation(ErrCalculator):
    def __init__(self, bits_per_unit: float, method: str = 'round') -> None:

        self._units_per_bit = 1/float(bits_per_unit)
        self._method = select_round_method(method)

    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        rounded_measurements = self._units_per_bit*self._method(
            err_basis/self._units_per_bit)
        return rounded_measurements - err_basis


class SysErrSaturation(ErrCalculator):
    def __init__(self,
                 meas_min: float,
                 meas_max: float) -> None:

        if meas_min > meas_max:
            raise ValueError("Minimum must be smaller than maximum for systematic error saturation")

        self._min = meas_min
        self._max = meas_max


    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        saturated = np.copy(err_basis)
        saturated[saturated > self._max] = self._max
        saturated[saturated < self._min] = self._min
        return saturated - err_basis



class SysErrCalibration(ErrCalculator):
    def __init__(self) -> None:
        pass

    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        # Need a calibration function

        return np.array([])



def select_round_method(method: str) -> Callable:

    if method == 'floor':
        return np.floor
    if method == 'ceil':
        return np.ceil
    return np.round


