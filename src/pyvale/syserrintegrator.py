'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
from pyvale.errorcalculator import ErrCalculator

# - A systematic error function returns an array of systematic errors, constant
# in time.
# - Output array is [sensor,component,time]
# - Need to loop over all functions so there will be an internal array that is
# [err_calc,sensor,component,time]
# - Then np.sum(axis=0)

class SysErrIntegrator():
    def __init__(self,
                 err_calcs: list[ErrCalculator],
                 meas_shape: tuple[int,int,int]) -> None:

        self._sys_errs = np.empty(meas_shape)
        self._sys_err_calcs = err_calcs
        self._meas_shape = meas_shape
        self.calc_all_sys_errs()

    def set_err_calcs(self, err_calcs: list[ErrCalculator]) -> None:

        self._sys_err_calcs = err_calcs
        self.calc_all_sys_errs()

    def add_err_calc(self, err_calc: ErrCalculator) -> None:

        self._sys_err_calcs.append(err_calc)
        self.calc_all_sys_errs()

    def calc_all_sys_errs(self) -> np.ndarray:

        n_erfs = len(self._sys_err_calcs)
        self._sys_errs = np.empty((n_erfs,
                            self._meas_shape[0],
                            self._meas_shape[1],
                            self._meas_shape[2]))

        for ii,ff in enumerate(self._sys_err_calcs):
            self._sys_errs[ii,:,:,:] = ff.calc_errs(self._meas_shape)

        return self._sys_errs

    def get_sys_errs_by_func(self) -> np.ndarray:

        return self._sys_errs

    def get_sys_errs_tot(self) -> np.ndarray:

        return np.sum(self._sys_errs,axis=0)








