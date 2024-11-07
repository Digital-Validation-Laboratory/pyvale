'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
from pyvale.sensordata import SensorData
from pyvale.errorcalculator import (IErrCalculator,
                                    EErrType,
                                    EErrDependence)
from pyvale.generatorsrandom import IGeneratorRandom


class ErrRandUniform(IErrCalculator):
    __slots__ = ("_low","_high","_rng","_err_dep")

    def __init__(self,
                 low: float,
                 high: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._low = low
        self._high = high
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

        rand_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_basis.shape)

        return (rand_errs,sens_data)


class ErrRandUnifPercent(IErrCalculator):
    __slots__ = ("_low","_high","_rng","_err_dep")

    def __init__(self,
                 low_percent: float,
                 high_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._low = low_percent
        self._high = high_percent
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

        norm_rand = self._rng.uniform(low=self._low/100,
                                    high=self._high/100,
                                    size=err_basis.shape)

        return (err_basis*norm_rand,sens_data)


class ErrRandNormal(IErrCalculator):
    __slots__ = ("_std","_rng","_err_dep")

    def __init__(self,
                 std: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._std = std
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
        rand_errs = self._rng.normal(loc=0.0,
                                    scale=self._std,
                                    size=err_basis.shape)

        return (rand_errs,sens_data)


class ErrRandNormPercent(IErrCalculator):
    __slots__ = ("_std","_rng","_err_dep")

    def __init__(self,
                 std_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._std = std_percent/100
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
