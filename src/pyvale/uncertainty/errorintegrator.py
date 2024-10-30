'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
from pyvale.uncertainty.errorcalculator import IErrCalculator


class ErrorIntegrator:

    def __init__(self,
                 err_calcs: list[IErrCalculator],
                 meas_shape: tuple[int,int,int]) -> None:

        self._err_calcs = err_calcs
        self._errs_by_func = np.zeros((len(err_calcs),
                                        meas_shape[0],
                                        meas_shape[1],
                                        meas_shape[2]))
        self._errs_tot = np.zeros(meas_shape)


    def set_err_calcs(self, err_calcs: list[IErrCalculator]) -> None:
        self._err_calcs = err_calcs


    def calc_errs_independent(self, err_basis: np.ndarray) -> np.ndarray:
        # NOTE: In this case the error basis is the ground truth and all errors
        # are calculated independently
        for ii,ff in enumerate(self._err_calcs):
            self._errs_by_func[ii,:,:,:] = ff.calc_errs(err_basis).error_array

        self._errs_tot = np.sum(self._errs_by_func,axis=0)
        return self._errs_tot

    def calc_errs_dependent(self, err_basis: np.ndarray) -> np.ndarray:
        # NOTE: In this case the error basis is the current value of all errors
        # summed (integrated) previously in the chain. So, the current error is
        # dependent on the accumulation of all previous errors.
        current_basis = np.copy(err_basis)
        for ii,ff in enumerate(self._err_calcs):
            self._errs_by_func[ii,:,:,:] = ff.calc_errs(current_basis).error_array
            current_basis = current_basis + self._errs_by_func[ii,:,:,:]

        self._errs_tot = np.sum(self._errs_by_func,axis=0)
        return self._errs_tot

    def get_errs_by_func(self) -> np.ndarray:
        return self._errs_by_func

    def get_errs_tot(self) -> np.ndarray:
        return self._errs_tot








