"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
import numpy as np


class IDriftCalculator(ABC):
    @abstractmethod
    def calc_drift(self,time_steps_by_sensor: np.ndarray) -> np.ndarray:
        pass


class DriftConstant(IDriftCalculator):
    def __init__(self, offset: float) -> None:
        self._offset = offset

    def calc_drift(self, time_steps_by_sensor: np.ndarray) -> np.ndarray:
        return self._offset*np.ones_like(time_steps_by_sensor)


class DriftLinear(IDriftCalculator):
    def __init__(self, slope: float, offset: float = 0.0) -> None:
        self._slope = slope
        self._offset = offset

    def calc_drift(self, time_steps_by_sensor: np.ndarray) -> np.ndarray:
        return self._slope*time_steps_by_sensor + self._offset