'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np

class ErrCalculator(ABC):
    @abstractmethod
    def calc_errs(self, meas_shape: tuple[int,...]) -> np.ndarray:
        pass

