"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
from pyvale.core.sensordata import SensorData
from pyvale.core.errorcalculator import (IErrCalculator,
                                         EErrType,
                                         EErrDependence)
from pyvale.core.generatorsrandom import IGeneratorRandom


class ErrRandUniform(IErrCalculator):
    """Sensor random error calculator based on uniform sampling of an interval
    specified by its upper and lower bound. This class implements the
    `IErrCalculator` interface.
    """
    __slots__ = ("low","high","rng","err_dep")

    def __init__(self,
                 low: float,
                 high: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        """Initialiser for `ErrRandUniform` class.

        Parameters
        ----------
        low : float
            Lower bound of the uniform random generator.
        high : float
            Upper bound of the uniform random generator.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.
        """
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)
        self.err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        """Gets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        For this class errors are calculated independently regardless.

        Returns
        -------
        EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        return self.err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        """Sets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        For this class errors are calculated independently regardless.

        Parameters
        ----------
        dependence : EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        self.err_dep = dependence

    def get_error_type(self) -> EErrType:
        """Gets the error type.

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        return EErrType.RANDOM

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        """Calculates the error array based on the size of the input

        Parameters
        ----------
        err_basis : np.ndarray
            Array of values with the same dimensions as the sensor measurement
            matrix.
        sens_data : SensorData
            A sensor data object

        Returns
        -------
        tuple[np.ndarray, SensorData]
            Tuple containing the calculated error array and pass through of the
            sensor data object as it is not modified by this class.
        """
        rand_errs = self.rng.uniform(low=self.low,
                                     high=self.high,
                                     size=err_basis.shape)

        return (rand_errs,sens_data)


class ErrRandUnifPercent(IErrCalculator):
    """Sensor random error calculator based on a percentage error taken from
    uniform sampling of an interval specified by its upper and lower bound (in
    percent). This class implements the `IErrCalculator` interface.
    """
    __slots__ = ("low","high","rng","err_dep")

    def __init__(self,
                 low_percent: float,
                 high_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        """_summary_

        Parameters
        ----------
        low_percent : float
            Lower percentage bound of the uniform random generator.
        high_percent : float
            Upper percentage bound of the uniform random generator.
        err_dep : EErrDependence, optional
            Error calculation dependence, by default EErrDependence.INDEPENDENT.
        seed : int | None, optional
            Optional seed for the random generator to allow for replicable
            behaviour, by default None.
        """
        self.low = low_percent/100
        self.high = high_percent/100
        self.rng = np.random.default_rng(seed)
        self.err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        """Gets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        Returns
        -------
        EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        return self.err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        """Sets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        Parameters
        ----------
        dependence : EErrDependence
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        self.err_dep = dependence

    def get_error_type(self) -> EErrType:
        """Gets the error type.

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        return EErrType.RANDOM

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        norm_rand = self.rng.uniform(low=self.low,
                                    high=self.high,
                                    size=err_basis.shape)

        return (err_basis*norm_rand,sens_data)


class ErrRandNormal(IErrCalculator):
    __slots__ = ("std","rng","err_dep")

    def __init__(self,
                 std: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        self.std = np.abs(std)
        self.rng = np.random.default_rng(seed)
        self.err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self.err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self.err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.RANDOM

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        rand_errs = self.rng.normal(loc=0.0,
                                    scale=self.std,
                                    size=err_basis.shape)

        return (rand_errs,sens_data)


class ErrRandNormPercent(IErrCalculator):
    __slots__ = ("_std","_rng","_err_dep")

    def __init__(self,
                 std_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._std = np.abs(std_percent)/100
        self._rng = np.random.default_rng(seed)
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.RANDOM

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        norm_rand = self._rng.normal(loc=0.0,
                                    scale=1.0,
                                    size=err_basis.shape)

        return (err_basis*self._std*norm_rand,sens_data)


class ErrRandGenerator(IErrCalculator):
    __slots__ = ("_generator","_err_dep")

    def __init__(self,
                 generator: IGeneratorRandom,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:
        self._generator = generator
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.RANDOM

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        rand_errs = self._generator.generate(size=err_basis.shape)

        return (rand_errs,sens_data)
