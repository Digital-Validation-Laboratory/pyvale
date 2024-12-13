'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import enum
from abc import ABC, abstractmethod
import numpy as np
from pyvale.core.sensordata import SensorData


class EErrType(enum.Enum):
    SYSTEMATIC = enum.auto()
    RANDOM = enum.auto()


class EErrDependence(enum.Enum):
    INDEPENDENT = enum.auto()
    DEPENDENT = enum.auto()


class IErrCalculator(ABC):
    @abstractmethod
    def get_error_type(self) -> EErrType:
        pass

    @abstractmethod
    def get_error_dep(self) -> EErrDependence:
        pass

    @abstractmethod
    def set_error_dep(self, dependence: EErrDependence) -> None:
        pass

    @abstractmethod
    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        pass




