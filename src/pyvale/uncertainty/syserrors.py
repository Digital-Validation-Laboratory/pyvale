'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
from pyvale.uncertainty.errorcalculator import IErrCalculator, ErrorData
from pyvale.uncertainty.randomgenerator import IRandomGenerator


class SysErrOffset(IErrCalculator):

    def __init__(self,
                 offset: float) -> None:
        self._offset = offset

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_data = ErrorData(error_array=
                             self._offset*np.ones(shape=err_basis.shape))
        return err_data


class SysErrOffsetPercent(IErrCalculator):

    def __init__(self,
                 offset_percent: float) -> None:
        self._offset_percent = offset_percent

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_data = ErrorData(error_array=self._offset_percent/100*err_basis*
                             np.ones(shape=err_basis.shape))
        return err_data


class SysErrUniform(IErrCalculator):

    def __init__(self,
                 low: float,
                 high: float,
                 seed: int | None = None) -> None:
        self._low = low
        self._high = high
        self._rng = np.random.default_rng(seed)

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        err_data = ErrorData(error_array=sys_errs)
        return err_data


class SysErrUnifPercent(IErrCalculator):

    def __init__(self,
                 low_percent: float,
                 high_percent: float,
                 seed: int | None = None) -> None:
        self._low = low_percent/100
        self._high = high_percent/100
        self._rng = np.random.default_rng(seed)

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.uniform(low=self._low,
                                    high=self._high,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        err_data = ErrorData(error_array=err_basis*sys_errs)
        return err_data



class SysErrNormal(IErrCalculator):

    def __init__(self,
                 std: float,
                 seed: int | None = None) -> None:
        self._std = std
        self._rng = np.random.default_rng(seed)

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.normal(loc=0.0,
                                    scale=self._std,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        err_data = ErrorData(error_array=sys_errs)
        return err_data


class SysErrNormPercent(IErrCalculator):

    def __init__(self,
                 std_percent: float,
                 seed: int | None = None) -> None:
        self._std = std_percent/100
        self._rng = np.random.default_rng(seed)

    def calc_errs(self,err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1
        sys_errs = self._rng.normal(loc=0.0,
                                    scale=self._std,
                                    size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        err_data = ErrorData(error_array=err_basis*sys_errs)
        return err_data


class SysErrGenerator(IErrCalculator):

    def __init__(self,
                 generator: IRandomGenerator) -> None:
        self._generator = generator

    def calc_errs(self,
                  err_basis: np.ndarray) -> ErrorData:

        err_shape = np.array(err_basis.shape)
        err_shape[-1] = 1

        sys_errs = self._generator.generate(size=err_shape)

        tile_shape = np.array(err_basis.shape)
        tile_shape[0:-1] = 1
        sys_errs = np.tile(sys_errs,tuple(tile_shape))

        err_data = ErrorData(error_array=sys_errs)
        return err_data




