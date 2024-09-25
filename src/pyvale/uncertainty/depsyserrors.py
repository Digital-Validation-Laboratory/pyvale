'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from typing import Callable
import numpy as np

from pyvale.uncertainty.errorcalculator import IErrCalculator, ErrorData


class SysErrRoundOff(IErrCalculator):
    def __init__(self, method: str = 'round', base: float = 1.0) -> None:

        self._base = base
        self._method = _select_round_method(method)


    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        rounded_measurements = self._base*self._method(err_basis/self._base)

        err_data = ErrorData(error_array=(rounded_measurements - err_basis))
        return err_data


class SysErrDigitisation(IErrCalculator):
    def __init__(self, bits_per_unit: float, method: str = 'round') -> None:

        self._units_per_bit = 1/float(bits_per_unit)
        self._method = _select_round_method(method)

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        rounded_measurements = self._units_per_bit*self._method(
            err_basis/self._units_per_bit)

        err_data = ErrorData(error_array=(rounded_measurements - err_basis))
        return err_data


class SysErrSaturation(IErrCalculator):
    def __init__(self,
                 meas_min: float,
                 meas_max: float) -> None:

        if meas_min > meas_max:
            raise ValueError("Minimum must be smaller than maximum for "+
                             "systematic error saturation")

        self._min = meas_min
        self._max = meas_max


    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        saturated = np.copy(err_basis)
        saturated[saturated > self._max] = self._max
        saturated[saturated < self._min] = self._min

        err_data = ErrorData(error_array=(saturated - err_basis))
        return err_data


class SysErrCalibration(IErrCalculator):
    def __init__(self, cal_func: Callable) -> None:
        self._cal_func = cal_func

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        # Need a calibration function
        err_data = ErrorData(error_array=np.array([]))
        return err_data


def _select_round_method(method: str) -> Callable:
    if method == 'floor':
        return np.floor
    if method == 'ceil':
        return np.ceil
    return np.round


