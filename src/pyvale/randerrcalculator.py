'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

from pyvale.errorcalculator import ErrCalculator


class RandErrUniform(ErrCalculator):

    def __init__(self,
                 low: float,
                 high: float) -> None:
        self._low = low
        self._high = high

    def calc_errs(self, meas_shape: tuple[int,...]) -> np.ndarray:
        rand_errs = np.random.default_rng().uniform(low=self._low,
                                                   high=self._high,
                                                   size=meas_shape)
        return rand_errs

class RandErrNormal(ErrCalculator):

    def __init__(self,
                 std: float) -> None:
        self._std = std

    def calc_errs(self, meas_shape: tuple[int,...]) -> np.ndarray:
        rand_errs = np.random.default_rng().normal(loc=0.0,
                                                   scale=self._std,
                                                   size=meas_shape)
        return rand_errs
