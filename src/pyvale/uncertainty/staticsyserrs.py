'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

from pyvale.uncertainty.errorcalculator import ErrCalculator


class SysErrDigitisation(ErrCalculator):
    def __init__(self, bits: int) -> None:
        self._bits = bits

    def calc_errs(self,
                  meas_shape: tuple[int,...],
                  truth_values: np.ndarray) -> np.ndarray:
        return np.array([])


class SysErrCalibration(ErrCalculator):
    def __init__(self) -> None:
        pass

    def calc_errs(self,
                  meas_shape: tuple[int,...],
                  truth_values: np.ndarray) -> np.ndarray:
        return np.array([])


class SysErrSaturation(ErrCalculator):
    def __init__(self,
                 meas_min: float,
                 meas_max: float) -> None:
        self._min = meas_min
        self._max = meas_max

    def calc_errs(self,
                  meas_shape: tuple[int,...],
                  truth_values: np.ndarray) -> np.ndarray:

        
        return np.array([])

