'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
import pyvista as pv

from pyvale.physics.field import IField
from pyvale.uncertainty.errorintegrator import ErrorIntegrator
from pyvale.sensors.sensordescriptor import SensorDescriptor
from pyvale.numerical.spatialintegrator import ISpatialIntegrator


class PointSensorArray:
    def __init__(self,
                 positions: np.ndarray,
                 field: IField,
                 sample_times: np.ndarray | None = None,
                 descriptor: SensorDescriptor | None = None,
                 area_avg: ISpatialIntegrator | None = None
                 ) -> None:

        self.positions = positions
        self.field = field

        self._sample_times = sample_times

        self.descriptor = SensorDescriptor()
        if descriptor is not None:
            self.descriptor = descriptor

        self._area_avg = area_avg

        self._truth = None
        self._measurements = None

        self._pre_syserr_integrator = None
        self._randerr_integrator = None
        self._post_syserr_integrator = None

    #---------------------------------------------------------------------------
    # accessors
    def set_sample_times(self, sample_times: np.ndarray | None) -> None:
        self._sample_times = sample_times

    def get_sample_times(self) -> np.ndarray:
        if self._sample_times is None:
            return self.field.get_time_steps()

        return self._sample_times

    def get_measurement_shape(self) -> tuple[int,int,int]:
        return (self.positions.shape[0],
                len(self.field.get_all_components()),
                self.get_sample_times().shape[0])

    #---------------------------------------------------------------------------
    # truth calculation from simulation
    def calc_truth_values(self) -> np.ndarray:
        if self._area_avg is None:
            return self.field.sample_field(self.positions,
                                            self._sample_times)

        return self._area_avg.calc_averages(self.positions,
                                                      self._sample_times)

    def get_truth_values(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth_values()

        return self._truth

    #---------------------------------------------------------------------------
    # pre / independent / truth-based  systematic errors
    def set_indep_sys_err_integrator(self,
                               err_int: ErrorIntegrator) -> None:
        self._pre_syserr_integrator = err_int


    def _calc_pre_systematic_errs(self) -> np.ndarray | None:
        if self._pre_syserr_integrator is None:
            return None

        self._pre_syserr_integrator.calc_errs_static(self.get_truth_values())
        return self._pre_syserr_integrator.get_errs_tot()


    def get_pre_systematic_errs(self) -> np.ndarray | None:
        if self._pre_syserr_integrator is None:
            return None

        return self._pre_syserr_integrator.get_errs_tot()

    #---------------------------------------------------------------------------
    # random errors
    def set_rand_err_integrator(self,
                                err_int: ErrorIntegrator) -> None:
        self._randerr_integrator = err_int


    def _calc_random_errs(self)-> np.ndarray | None:
        if self._randerr_integrator is None:
            return None

        self._randerr_integrator.calc_errs_static(self.get_truth_values())
        return self._randerr_integrator.get_errs_tot()


    def get_random_errs(self) -> np.ndarray | None:
        if self._randerr_integrator is None:
            return None

        return self._randerr_integrator.get_errs_tot()

    #---------------------------------------------------------------------------
    # post / coupled / measurement based systematic errors
    def set_dep_sys_err_integrator(self,
                               err_int: ErrorIntegrator) -> None:
        self._post_syserr_integrator = err_int


    def _calc_dep_systematic_errs(self, measurements: np.ndarray
                                   ) -> np.ndarray | None:
        if self._post_syserr_integrator is None:
            return None

        self._post_syserr_integrator.calc_errs_recursive(measurements)
        return self._post_syserr_integrator.get_errs_tot()


    def get_dep_systematic_errs(self) -> np.ndarray | None:
        if self._post_syserr_integrator is None:
            return None

        return self._post_syserr_integrator.get_errs_tot()

    #---------------------------------------------------------------------------
    # measurements
    def calc_measurements(self) -> np.ndarray:
        measurements = self.get_truth_values()

        indep_sys_errs = self._calc_pre_systematic_errs()
        if indep_sys_errs is not None:
            measurements = measurements + indep_sys_errs

        rand_errs = self._calc_random_errs()
        if rand_errs is not None:
            measurements = measurements + rand_errs

        dep_sys_errs = self._calc_dep_systematic_errs(measurements)
        if dep_sys_errs is not None:
            measurements = measurements + dep_sys_errs

        self._measurements = measurements
        return self._measurements


    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.calc_measurements()

        return self._measurements

    #---------------------------------------------------------------------------
    # visualisation tools
    def get_visualiser(self) -> pv.PolyData:
        pv_data = pv.PolyData(self.positions)
        return pv_data

