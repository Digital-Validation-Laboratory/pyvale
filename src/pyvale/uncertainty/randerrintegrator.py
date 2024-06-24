'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
from pyvale.uncertainty.errorcalculator import ErrCalculator


class RandErrIntegrator():
    def __init__(self,
                 err_calcs: list[ErrCalculator],
                 meas_shape: tuple[int,int,int]) -> None:

        self._rand_errs = np.empty(meas_shape)
        self._rand_err_calcs = err_calcs
        self._meas_shape = meas_shape
        self.calc_all_rand_errs()

    def set_err_calcs(self, err_calcs: list[ErrCalculator]) -> None:
        self._rand_err_calcs = err_calcs


    def add_err_calc(self, err_calc: ErrCalculator) -> None:
        self._rand_err_calcs.append(err_calc)


    def calc_all_rand_errs(self) -> np.ndarray:

        n_erfs = len(self._rand_err_calcs)
        self._rand_errs = np.empty((n_erfs,
                            self._meas_shape[0],
                            self._meas_shape[1],
                            self._meas_shape[2]))

        for ii,ff in enumerate(self._rand_err_calcs):
            self._rand_errs[ii,:,:,:] = ff.calc_errs(self._meas_shape)

        return self._rand_errs

    def get_rand_errs_by_func(self) -> np.ndarray:
        self.calc_all_rand_errs()
        return self._rand_errs

    def get_rand_errs_tot(self) -> np.ndarray:
        self.calc_all_rand_errs()
        return np.sum(self._rand_errs,axis=0)








