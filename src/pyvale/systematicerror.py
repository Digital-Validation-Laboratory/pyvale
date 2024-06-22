'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Callable
from functools import partial

import numpy as np

# - A systematic error function returns an array of systematic errors, constant
# in time.
# - Output array is [sensor,component,time]
# - Need to loop over all functions so there will be an internal array that is
# [err_func,sensor,component,time]
# - Then np.sum(axis=0)

class SystematicErrorGenerator():
    def __init__(self,
                 err_funcs: list[Callable],
                 meas_shape: tuple[int,int,int]) -> None:

        self._sys_errs = np.array(())
        self._sys_err_funcs = err_funcs
        self._meas_shape = meas_shape

    def set_err_funcs(self, err_funcs: list[Callable]) -> None:

        self._sys_err_funcs = err_funcs
        self.calc_sys_errs()

    def add_err_func(self, err_func: Callable) -> None:

        self._sys_err_funcs.append(err_func)
        self.calc_sys_errs()

    def calc_sys_errs(self) -> np.ndarray | None:

        n_erfs = len(self._sys_err_funcs)
        self._sys_errs = np.empty((n_erfs,
                                   self._meas_shape[0],
                                   self._meas_shape[1],
                                   self._meas_shape[2]))

        for ff in self._sys_err_funcs:
            self._sys_errs[ff,:,:,:] = ff(self._meas_shape)

        return self._sys_errs

    def get_sys_errs_by_func(self) -> np.ndarray | None:
        return self._sys_errs

    def get_sys_errs_tot(self) -> np.ndarray | None:
        return np.squeeze(np.sum(self._sys_errs,axis=0))


def build_uniform_err_func(meas_shape: tuple[int,int,int],
                           low: float,
                           high: float) -> Callable:

    def sys_err_func(size: tuple,low: float, high: float) -> np.ndarray:
        sys_errs = np.random.default_rng().uniform(low=low,
                                                high=high,
                                                size=(meas_shape[0],1,1))
        sys_errs = np.tile(sys_errs,(1,1,meas_shape[2]))
        return sys_errs

    return partial(sys_err_func,low=low,high=high)






