'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np

class IErrCalculator(ABC):
    @abstractmethod
    def calc_errs(self,err_basis: np.ndarray) -> np.ndarray:
        pass




