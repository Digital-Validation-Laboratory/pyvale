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
                                                EErrType,
                                                EErrDependence)
from pyvale.sensors.sensordata import SensorData

@dataclass(slots=True)
class ErrorIntegrationOpts:
    force_dependence: bool = False
    store_errs_by_func: bool = False


class ErrorIntegrator:
    __slots__ = ("_err_chain","_meas_shape","_errs_by_chain",
                 "_errs_systematic","_errs_random","_errs_total",
                 "_sens_data_by_chain","_err_int_opts")

    def __init__(self,
                 err_chain: list[IErrCalculator],
                 meas_shape: tuple[int,int,int],
                 err_int_opts: ErrorIntegrationOpts | None = None) -> None:

        if err_int_opts is None:
            self._err_int_opts = ErrorIntegrationOpts()
        else:
            self._err_int_opts = err_int_opts

        self.set_error_chain(err_chain)
        self._meas_shape = meas_shape

        self._sens_data_by_chain = []

        if self._err_int_opts.store_errs_by_func:
            self._errs_by_chain = np.zeros((len(self._err_chain),)+ \
                                               self._meas_shape)
        else:
            self._errs_by_chain = None

        self._errs_systematic = np.zeros(meas_shape)
        self._errs_random = np.zeros(meas_shape)
        self._errs_total = np.zeros(meas_shape)


    def set_error_chain(self, err_chain: list[IErrCalculator]) -> None:
        self._err_chain = err_chain

        if self._err_int_opts.force_dependence:
            for ee in self._err_chain:
                ee.set_error_dep(EErrDependence.DEPENDENT)


    def calc_errors_from_chain(self, truth: np.ndarray) -> np.ndarray:
        if self._err_int_opts.store_errs_by_func:
            return self._calc_errors_store_by_func(truth)

        return self._calc_errors_mem_eff(truth)


    def _calc_errors_store_by_func(self, truth: np.ndarray) -> np.ndarray:
        accumulated_error = np.zeros_like(truth)
        self._errs_by_chain = np.zeros((len(self._err_chain),) + \
                                           self._meas_shape)

        for ii,ee in enumerate(self._err_chain):

            if ee.get_error_dep() == EErrDependence.DEPENDENT:
                (error_array,sens_data) = ee.calc_errs(truth+accumulated_error)
            else:
                (error_array,sens_data) = ee.calc_errs(truth)

            self._sens_data_by_chain.append(sens_data)

            if ee.get_error_type() == EErrType.SYSTEMATIC:
                self._errs_systematic = self._errs_systematic + error_array
            else:
                self._errs_random = self._errs_random + error_array

            accumulated_error = accumulated_error + error_array
            self._errs_by_chain[ii,:,:,:] = error_array

        self._errs_total = accumulated_error
        return self._errs_total


    def _calc_errors_mem_eff(self, truth: np.ndarray) -> np.ndarray:
        accumulated_error = np.zeros_like(truth)

        for ee in self._err_chain:

            if ee.get_error_dep() == EErrDependence.DEPENDENT:
                (error_array,sens_data) = ee.calc_errs(truth+accumulated_error)
            else:
                (error_array,sens_data) = ee.calc_errs(truth)

            self._sens_data_by_chain.append(sens_data)

            if ee.get_error_type() == EErrType.SYSTEMATIC:
                self._errs_systematic = self._errs_systematic + error_array
            else:
                self._errs_random = self._errs_random + error_array

            accumulated_error = accumulated_error + error_array

        self._errs_total = accumulated_error
        return self._errs_total


    def get_errs_by_chain(self) -> np.ndarray | None:
        return self._errs_by_chain

    def get_sens_data_by_chain(self) -> list[SensorData]:
        return self._sens_data_by_chain

    def get_errs_systematic(self) -> np.ndarray:
        return self._errs_systematic

    def get_errs_random(self) -> np.ndarray:
        return self._errs_random

    def get_errs_total(self) -> np.ndarray:
        return self._errs_total











