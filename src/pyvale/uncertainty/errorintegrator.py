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

        self._err_calcs = err_calcs
        self._meas_shape = meas_shape
        self._errs_by_func = np.zeros((len(err_calcs),
                            self._meas_shape[0],
                            self._meas_shape[1],
                            self._meas_shape[2]))


    def set_err_calcs(self, err_calcs: list[ErrCalculator]) -> None:
        self._err_calcs = err_calcs


    def calc_all_errs(self, err_basis: np.ndarray) -> np.ndarray:
        for ii,ff in enumerate(self._err_calcs):
            self._errs_by_func[ii,:,:,:] = ff.calc_errs(self._meas_shape,
                                                        err_basis)
        return self._errs_by_func


    def get_errs_by_func(self) -> np.ndarray:
        return self._errs_by_func


    def get_errs_tot(self) -> np.ndarray:
        return np.sum(self._errs_by_func,axis=0)








