'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ErrorData:
    error_array: np.ndarray | None = None
    positions: np.ndarray | None = None
    angles: tuple[Rotation,...] | None = None
    time_steps: np.ndarray | None = None


class IErrCalculator(ABC):
    @abstractmethod
    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:
        pass




