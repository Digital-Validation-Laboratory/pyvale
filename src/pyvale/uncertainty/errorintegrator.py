'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from dataclasses import dataclass
import numpy as np
from pyvale.uncertainty.errorcalculator import (IErrCalculator,
                                                EErrorType,
                                                EErrorCalc)
#from pyvale.sensors.pointsensorarray import SensorData

@dataclass(slots=True)
class ErrorIntegrationOpts:
    force_dependence: bool = False
    store_errs_by_func: bool = False


class ErrorIntegrator:
    __slots__ = ("err_chain","meas_shape","store_errs_by_func","_errs_by_func",
                 "_errs_systematic","_errs_random","_errs_total")

    def __init__(self,
                 err_chain: list[IErrCalculator],
                 meas_shape: tuple[int,int,int],
                 store_errs_by_func: bool = True) -> None:

        self.err_chain = err_chain
        self.meas_shape = meas_shape
        self.store_errs_by_func = store_errs_by_func

        if store_errs_by_func:
            self._errs_by_func = np.zeros((len(self.err_chain),)+ \
                                               self.meas_shape)
        else:
            self._errs_by_func = None

        self._errs_systematic = np.zeros(meas_shape)
        self._errs_random = np.zeros(meas_shape)
        self._errs_total = np.zeros(meas_shape)


    def set_err_calcs(self, err_calcs: list[IErrCalculator]) -> None:
        self.err_chain = err_calcs


    def calc_errors_from_chain(self, truth: np.ndarray) -> np.ndarray:
        if self.store_errs_by_func:
            return self._calc_errors_store_by_func(truth)

        return self._calc_errors_mem_eff(truth)


    def _calc_errors_store_by_func(self, truth: np.ndarray) -> np.ndarray:
        accumulated_error = np.zeros_like(truth)
        self._errs_by_func = np.zeros((len(self.err_chain),) + \
                                           self.meas_shape)

        for ii,ee in enumerate(self.err_chain):

            if ee.get_error_calc() == EErrorCalc.DEPENDENT:
                self._errs_by_func[ii,:,:,:] = ee.calc_errs(truth+accumulated_error).error_array
            else:
                self._errs_by_func[ii,:,:,:] = ee.calc_errs(truth).error_array

            if ee.get_error_type() == EErrorType.SYSTEMATIC:
                self._errs_systematic = self._errs_systematic + \
                                        self._errs_by_func[ii,:,:,:]
            else:
                self._errs_random = self._errs_random + \
                                    self._errs_by_func[ii,:,:,:]

            accumulated_error = accumulated_error + self._errs_by_func[ii,:,:,:]

        self._errs_total = accumulated_error
        return self._errs_total


    def _calc_errors_mem_eff(self, truth: np.ndarray) -> np.ndarray:
        accumulated_error = np.zeros_like(truth)

        for ee in self.err_chain:
            current_errs = np.zeros(self.meas_shape)

            if ee.get_error_calc() == EErrorCalc.DEPENDENT:
                current_errs = ee.calc_errs(truth+accumulated_error).error_array
            else:
                current_errs = ee.calc_errs(truth).error_array

            if ee.get_error_type() == EErrorType.SYSTEMATIC:
                self._errs_systematic = self._errs_systematic + \
                                        current_errs
            else:
                self._errs_random = self._errs_random + \
                                    current_errs

            accumulated_error = accumulated_error+current_errs

        self._errs_total = accumulated_error
        return self._errs_total


    def get_errs_by_func(self) -> np.ndarray | None:
        return self._errs_by_func

    def get_errs_systematic(self) -> np.ndarray:
        return self._errs_systematic

    def get_errs_random(self) -> np.ndarray:
        return self._errs_random

    def get_errs_total(self) -> np.ndarray:
        return self._errs_total








