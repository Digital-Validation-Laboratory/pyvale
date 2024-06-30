'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

from pyvale.uncertainty.errorcalculator import ErrCalculator


class SysErrRoundOff(ErrCalculator):
    def __init__(self, method: str = 'round') -> None:

        if method == 'floor':
            self._method = np.floor
        elif method == 'ceil':
            self._method = np.ceil
        else:
            self._method = np.round

    def calc_errs(self,
                  meas_shape: tuple[int,...],
                  err_basis: np.ndarray) -> np.ndarray:

        rounded_measurements = self._method(err_basis)
        return rounded_measurements - err_basis



class SysErrCalibration(ErrCalculator):
    def __init__(self) -> None:
        pass

    def calc_errs(self,
                  meas_shape: tuple[int,...],
                  err_basis: np.ndarray) -> np.ndarray:

        # Need a calibration function

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

