'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np

class SysErrCalculator(ABC):

    @abstractmethod
    def calc_sys_errs(self, meas_shape: tuple[int,...]) -> np.ndarray:
        pass


class SysErrUniform(SysErrCalculator):

    def __init__(self,
                 low: float,
                 high: float) -> None:
        self._low = low
        self._high = high

    def calc_sys_errs(self, meas_shape: tuple[int,...]) -> np.ndarray:

            sys_errs = np.random.default_rng().uniform(low=self._low,
                                                    high=self._high,
                                                    size=(meas_shape[0],1,1))
            sys_errs = np.tile(sys_errs,(1,1,meas_shape[1]))

            return sys_errs




