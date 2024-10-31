'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(slots=True)
class ErrorData:
    error_array: np.ndarray | None = None
    positions: np.ndarray | None = None
    angles: tuple[Rotation,...] | None = None
    time_steps: np.ndarray | None = None


class EErrorType(enum.Enum):
    SYSTEMATIC = enum.auto()
    RANDOM = enum.auto()

class EErrorCalc(enum.Enum):
    INDEPENDENT = enum.auto()
    DEPENDENT = enum.auto()


class IErrCalculator(ABC):
    @abstractmethod
    def get_error_type(self) -> EErrorType:
        pass

    @abstractmethod
    def get_error_calc(self) -> EErrorCalc:
        pass

    @abstractmethod
    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:
        pass




