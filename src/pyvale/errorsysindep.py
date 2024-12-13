'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Callable
import numpy as np
from pyvale.errorcalculator import (IErrCalculator,
                                    EErrType,
                                    EErrDependence)
from pyvale.generatorsrandom import IGeneratorRandom
from pyvale.sensordata import SensorData


class ErrSysOffset(IErrCalculator):
    __slots__ = ("_offset","_err_dep")

    def __init__(self,
                 offset: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:
        self._offset = offset
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        return (self._offset*np.ones(shape=err_basis.shape),sens_data)


class ErrSysOffsetPercent(IErrCalculator):
    __slots__ = ("_offset_percent","_err_dep")

    def __init__(self,
                 offset_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:
        self._offset_percent = offset_percent
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        return (self._offset_percent/100 *
                err_basis *
                np.ones(shape=err_basis.shape),
                None)


class ErrSysUniform(IErrCalculator):
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
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (sys_errs,sens_data)


class ErrSysUniformPercent(IErrCalculator):
    __slots__ = ("_low","_high","_rng","_err_dep")

    def __init__(self,
                 low_percent: float,
                 high_percent: float,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._low = low_percent/100
        self._high = high_percent/100
        self._rng = np.random.default_rng(seed)
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (err_basis*sys_errs,sens_data)


class ErrSysNormal(IErrCalculator):
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
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.normal(loc=0.0,
                                    scale=self._std,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (sys_errs,sens_data)


class ErrSysNormPercent(IErrCalculator):
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
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.normal(loc=0.0,
                                    scale=self._std,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (err_basis*sys_errs,sens_data)


class ErrSysGenerator(IErrCalculator):
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
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1

        sys_errs = self._generator.generate(size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return (sys_errs,sens_data)


class ErrSysCalibration(IErrCalculator):
    __slots__ = ("_assumed_cali","_truth_calib","_cal_range","_n_cal_divs",
                 "_err_dep","_truth_calc_table")

    def __init__(self,
                 assumed_calib: Callable,
                 truth_calib: Callable,
                 cal_range: tuple[float,float],
                 n_cal_divs: int = 10000,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._assumed_calib = assumed_calib
        self._truth_calib = truth_calib
        self._cal_range = cal_range
        self._n_cal_divs = n_cal_divs
        self._err_dep = err_dep

        self._truth_cal_table = np.zeros((n_cal_divs,2))
        self._truth_cal_table[:,0] = np.linspace(cal_range[0],
                                                cal_range[1],
                                                n_cal_divs)
        self._truth_cal_table[:,1] = self._truth_calib(self._truth_cal_table[:,0])

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        # shape=(n_sens,n_comps,n_time_steps)
        signal_from_field = np.interp(err_basis,
                                    self._truth_cal_table[:,1],
                                    self._truth_cal_table[:,0])
        # shape=(n_sens,n_comps,n_time_steps)
        field_from_assumed_calib = self._assumed_calib(signal_from_field)

        sys_errs = field_from_assumed_calib - err_basis

        return (sys_errs,sens_data)




