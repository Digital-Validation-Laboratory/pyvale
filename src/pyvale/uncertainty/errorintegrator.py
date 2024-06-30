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
        self._errs_by_func = np.zeros((len(err_calcs),
                                        meas_shape[0],
                                        meas_shape[1],
                                        meas_shape[2]))
        self._errs_tot = np.zeros(meas_shape)


    def set_err_calcs(self, err_calcs: list[ErrCalculator]) -> None:
        self._err_calcs = err_calcs


    def calc_errs_static(self, err_basis: np.ndarray) -> np.ndarray:

        for ii,ff in enumerate(self._err_calcs):
            self._errs_by_func[ii,:,:,:] = ff.calc_errs(err_basis)

        self._errs_tot = np.sum(self._errs_by_func,axis=0)
        return self._errs_tot

    def calc_errs_recursive(self, err_basis: np.ndarray) -> np.ndarray:

        current_basis = np.copy(err_basis)
        for ii,ff in enumerate(self._err_calcs):
            self._errs_by_func[ii,:,:,:] = ff.calc_errs(current_basis)
            current_basis = current_basis + self._errs_by_func[ii,:,:,:]

        self._errs_tot = np.sum(self._errs_by_func,axis=0)
        return self._errs_tot

    def get_errs_by_func(self) -> np.ndarray:
        return self._errs_by_func

    def get_errs_tot(self) -> np.ndarray:
        return self._errs_tot








