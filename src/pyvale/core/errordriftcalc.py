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
    """Interface (abstract base class) for applying a function to cause a sensor
    array measurement to drift (normally over time). The initialiser for the
    concrete implementation of this class specifies the function parameters and
    a unified method is provided to calculate the drift based on the input.
    """

    @abstractmethod
    def calc_drift(self, drift_var: np.ndarray) -> np.ndarray:
        """Abstract method. Used to calculate the drift function based on the

        Parameters
        ----------
        time_steps : np.ndarray
            Array of time steps at which the sensors take measurements to be
            used to calculate the drift.

        Returns
        -------
        np.ndarray
            _description_
        """
        pass


class DriftConstant(IDriftCalculator):
    """_summary_

    Implements the IDriftCalculator interface.
    """
    def __init__(self, offset: float) -> None:
        self._offset = offset

    def calc_drift(self, time_steps: np.ndarray) -> np.ndarray:
        return self._offset*np.ones_like(time_steps)


class DriftLinear(IDriftCalculator):
    """_summary_

    Implements the IDriftCalculator interface.
    """
    
    def __init__(self, slope: float, offset: float = 0.0) -> None:
        self._slope = slope
        self._offset = offset

    def calc_drift(self, time_steps: np.ndarray) -> np.ndarray:
        return self._slope*time_steps + self._offset