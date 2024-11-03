'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import enum
from typing import Callable
import numpy as np
from pyvale.sensors.sensordata import SensorData
from pyvale.uncertainty.errorcalculator import (IErrCalculator,
                                                EErrType,
                                                EErrDependence)


class ERoundMethod(enum.Enum):
    ROUND = enum.auto()
    FLOOR = enum.auto()
    CEIL = enum.auto()


def _select_round_method(method: ERoundMethod) -> Callable:
    if method == ERoundMethod.FLOOR:
        return np.floor
    if method == ERoundMethod.CEIL:
        return np.ceil
    return np.round


class SysErrRoundOff(IErrCalculator):
    __slots__ = ("_base","_method","_err_dep")

    def __init__(self,
                 method: ERoundMethod = ERoundMethod.ROUND,
                 base: float = 1.0,
                 err_dep: EErrDependence = EErrDependence.DEPENDENT) -> None:

        self._base = base
        self._method = _select_round_method(method)
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:

        rounded_measurements = self._base*self._method(err_basis/self._base)

        return (rounded_measurements - err_basis,None)


class SysErrDigitisation(IErrCalculator):
    __slots__ = ("_units_per_bit","_method","_err_dep")

    def __init__(self,
                 bits_per_unit: float,
                 method: ERoundMethod = ERoundMethod.ROUND,
                 err_dep: EErrDependence = EErrDependence.DEPENDENT) -> None:

        self._units_per_bit = 1/float(bits_per_unit)
        self._method = _select_round_method(method)
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:

        rounded_measurements = self._units_per_bit*self._method(
            err_basis/self._units_per_bit)

        return (rounded_measurements - err_basis, None)


class SysErrSaturation(IErrCalculator):
    __slots__ = ("_min","_max","_err_dep")

    def __init__(self,
                 meas_min: float,
                 meas_max: float,
                 err_dep: EErrDependence = EErrDependence.DEPENDENT) -> None:

        if meas_min > meas_max:
            raise ValueError("Minimum must be smaller than maximum for "+
                             "systematic error saturation")

        self._min = meas_min
        self._max = meas_max
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:

        saturated = np.copy(err_basis)
        saturated[saturated > self._max] = self._max
        saturated[saturated < self._min] = self._min

        return (saturated - err_basis,None)




