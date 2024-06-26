'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
from pyvale.uncertainty.errorcalculator import ErrCalculator


class ErrorIntegrator():
    def __init__(self,
                 err_calcs: list[ErrCalculator],
                 meas_shape: tuple[int,int,int]) -> None:

        self._errs_tot = np.empty(meas_shape)
        self._err_calcs = err_calcs
        self._meas_shape = meas_shape
        self.calc_all_errs()

    def set_err_calcs(self, err_calcs: list[ErrCalculator]) -> None:
        self._err_calcs = err_calcs


    def add_err_calc(self, err_calc: ErrCalculator) -> None:
        self._err_calcs.append(err_calc)


    def calc_all_errs(self) -> np.ndarray:

        n_erfs = len(self._err_calcs)
        self._errs_tot = np.empty((n_erfs,
                            self._meas_shape[0],
                            self._meas_shape[1],
                            self._meas_shape[2]))

        for ii,ff in enumerate(self._err_calcs):
            self._errs_tot[ii,:,:,:] = ff.calc_errs(self._meas_shape)

        return self._errs_tot


    def get_errs_by_func(self) -> np.ndarray:
        return self._errs_tot


    def get_errs_tot(self) -> np.ndarray:
        return np.sum(self._errs_tot,axis=0)








