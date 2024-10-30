'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from typing import Callable
import numpy as np
from pyvale.uncertainty.errorcalculator import (IErrCalculator,
                                                ErrorData,
                                                EErrorType,
                                                EErrorCalc)
from pyvale.uncertainty.randomgenerator import IGeneratorRandom


class SysErrOffset(IErrCalculator):

    def __init__(self,
                 offset: float,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:
        self._offset = offset
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        return ErrorData(error_array=
                             self._offset*np.ones(shape=err_basis.shape))


class SysErrOffsetPercent(IErrCalculator):

    def __init__(self,
                 offset_percent: float,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:
        self._offset_percent = offset_percent
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        return ErrorData(error_array=self._offset_percent/100*err_basis*
                             np.ones(shape=err_basis.shape))


class SysErrUniform(IErrCalculator):

    def __init__(self,
                 low: float,
                 high: float,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return ErrorData(error_array=sys_errs)


class SysErrUniformPercent(IErrCalculator):
    def __init__(self,
                 low_percent: float,
                 high_percent: float,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._low = low_percent/100
        self._high = high_percent/100
        self._rng = np.random.default_rng(seed)
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return ErrorData(error_array=err_basis*sys_errs)


class SysErrNormal(IErrCalculator):
    def __init__(self,
                 std: float,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._std = std
        self._rng = np.random.default_rng(seed)
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.normal(loc=0.0,
                                    scale=self._std,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return ErrorData(error_array=sys_errs)


class SysErrNormPercent(IErrCalculator):
    def __init__(self,
                 std_percent: float,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT,
                 seed: int | None = None) -> None:
        self._std = std_percent/100
        self._rng = np.random.default_rng(seed)
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.normal(loc=0.0,
                                    scale=self._std,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return ErrorData(error_array=err_basis*sys_errs)


class SysErrGenerator(IErrCalculator):
    def __init__(self,
                 generator: IGeneratorRandom,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:
        self._generator = generator
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1

        sys_errs = self._generator.generate(size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        return ErrorData(error_array=sys_errs)


class SysErrCalibration(IErrCalculator):
    def __init__(self,
                 assumed_calib: Callable,
                 truth_calib: Callable,
                 cal_range: tuple[float,float],
                 n_cal_divs: int = 10000,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:

        self._assumed_calib = assumed_calib
        self._truth_calib = truth_calib
        self._cal_range = cal_range
        self._n_cal_divs = n_cal_divs
        self._err_calc = err_calc

        self._truth_cal_table = np.zeros((n_cal_divs,2))
        self._truth_cal_table[:,0] = np.linspace(cal_range[0],
                                                cal_range[1],
                                                n_cal_divs)
        self._truth_cal_table[:,1] = self._truth_calib(self._truth_cal_table[:,0])

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self,
                  err_basis: np.ndarray) -> ErrorData:

        # shape=(n_sens,n_comps,n_time_steps)
        signal_from_field = np.interp(err_basis,
                                    self._truth_cal_table[:,1],
                                    self._truth_cal_table[:,0])
        # shape=(n_sens,n_comps,n_time_steps)
        field_from_assumed_calib = self._assumed_calib(signal_from_field)

        sys_errs = field_from_assumed_calib - err_basis

        return ErrorData(error_array=sys_errs)




