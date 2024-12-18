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
    """Enumeration defining the error type for separation of error types for
    later analysis.

    SYSTEMATIC:
        Also known as an epistemic error and is due to a lack of
        knowledge. Common examples include spatial or temporal averaging,
        digitisation / round off error and calibration errors.

    RANDOM:
        Also known as aleatory error and is generally a result of sensor
        noise.
    """
    SYSTEMATIC = enum.auto()
    RANDOM = enum.auto()


class EErrDependence(enum.Enum):
    """Enumeration defining error dependence:

    INDEPENDENT:
        Errors are calculated based on the ground truth sensor values
        interpolated from the input simulation.

    DEPENDENT:
        Errors are calculated based on the accumulated sensor reading due
        to all preceeding errors in the chain.
    """
    INDEPENDENT = enum.auto()
    DEPENDENT = enum.auto()


class IErrCalculator(ABC):
    @abstractmethod
    def get_error_type(self) -> EErrType:
        """Abstract method for getting the error type.

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        pass

    @abstractmethod
    def get_error_dep(self) -> EErrDependence:
        """Abstract method for getting the error dependence.

        Returns
        -------
        EErrDependence
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        pass

    @abstractmethod
    def set_error_dep(self, dependence: EErrDependence) -> None:
        """Abstract method for setting the error dependence.

        Parameters
        ----------
        dependence : EErrDependence
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """

    @abstractmethod
    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        """Abstract method that calculates the error array based on the input
        err_basis array. The output error array will be the same shape as the
        input err_basis array.

        Parameters
        ----------
        err_basis : np.ndarray
            Used as the base array for calculating the returned error
        sens_data : SensorData
            Sensor data object holding the current sensor state before applying
            this error calculation.

        Returns
        -------
        tuple[np.ndarray, SensorData]
            Tuple containing the error array from this calculator and a
            SensorData object with the current accumulated sensor state starting
            from the nominal state up to and including this error calculator in
            the error chain.
        """
        pass




