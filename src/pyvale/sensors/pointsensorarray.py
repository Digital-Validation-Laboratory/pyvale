'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv

from pyvale.physics.field import IField
from pyvale.uncertainty.errorintegrator import ErrorIntegrator
from pyvale.sensors.sensordescriptor import SensorDescriptor
from pyvale.numerical.spatialintegrator import ISpatialAverager


@dataclass
class SensorData:
    positions: np.ndarray
    sample_times: np.ndarray | None = None
    area_averager: ISpatialAverager | None = None
    angles: tuple[Rotation,...] | None = None


class PointSensorArray:

    def __init__(self,
                 sensor_array_data: SensorData,
                 field: IField,
                 descriptor: SensorDescriptor | None = None,
                 ) -> None:

        self.sensor_data = sensor_array_data
        self.field = field

        self.descriptor = SensorDescriptor()
        if descriptor is not None:
            self.descriptor = descriptor


        self._truth = None
        self._measurements = None

        self._syserr_integrator_independent = None
        self._randerr_integrator = None
        self._syserr_integrator_dependent = None

    #---------------------------------------------------------------------------
    # accessors
    def get_sample_times(self) -> np.ndarray:
        if self.sensor_data.sample_times is None:
            return self.field.get_time_steps()

        return self.sensor_data.sample_times

    def get_measurement_shape(self) -> tuple[int,int,int]:
        return (self.sensor_data.positions.shape[0],
                len(self.field.get_all_components()),
                self.get_sample_times().shape[0])

    #---------------------------------------------------------------------------
    # truth calculation from simulation
    def calc_truth_values(self) -> np.ndarray:
        if self.sensor_data.area_averager is None:
            return self.field.sample_field(self.sensor_data.positions,
                                           self.sensor_data.sample_times,
                                           self.sensor_data.angles)

        return self.sensor_data.area_averager.calc_averages(
                        self.sensor_data.positions,
                        self.sensor_data.sample_times)

    def get_truth_values(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth_values()

        return self._truth

    #---------------------------------------------------------------------------
    # independent systematic errors calculated based on ground truth
    def set_systematic_err_integrator_independent(self,
                               err_int: ErrorIntegrator) -> None:
        self._syserr_integrator_independent = err_int


    def _calc_systematic_errs_independent(self) -> np.ndarray | None:
        if self._syserr_integrator_independent is None:
            return None

        self._syserr_integrator_independent.calc_errs_independent(
                                                self.get_truth_values())
        return self._syserr_integrator_independent.get_errs_tot()


    def get_systematic_errs_independent(self) -> np.ndarray | None:
        if self._syserr_integrator_independent is None:
            return None

        return self._syserr_integrator_independent.get_errs_tot()

    #---------------------------------------------------------------------------
    # random errors
    def set_random_err_integrator(self,
                                err_int: ErrorIntegrator) -> None:
        self._randerr_integrator = err_int


    def _calc_random_errs(self)-> np.ndarray | None:
        if self._randerr_integrator is None:
            return None

        self._randerr_integrator.calc_errs_independent(self.get_truth_values())
        return self._randerr_integrator.get_errs_tot()


    def get_random_errs(self) -> np.ndarray | None:
        if self._randerr_integrator is None:
            return None

        return self._randerr_integrator.get_errs_tot()

    #---------------------------------------------------------------------------
    # dependent systematic errors calculated based on integrated error
    def set_systematic_err_integrator_dependent(self,
                               err_int: ErrorIntegrator) -> None:
        self._syserr_integrator_dependent = err_int


    def _calc_systematic_errs_dependent(self, measurements: np.ndarray
                                   ) -> np.ndarray | None:
        if self._syserr_integrator_dependent is None:
            return None

        self._syserr_integrator_dependent.calc_errs_dependent(measurements)
        return self._syserr_integrator_dependent.get_errs_tot()


    def get_systematic_errs_dependent(self) -> np.ndarray | None:
        if self._syserr_integrator_dependent is None:
            return None

        return self._syserr_integrator_dependent.get_errs_tot()

    #---------------------------------------------------------------------------
    # measurements
    def calc_measurements(self) -> np.ndarray:
        measurements = self.get_truth_values()

        indep_sys_errs = self._calc_systematic_errs_independent()
        if indep_sys_errs is not None:
            measurements = measurements + indep_sys_errs

        rand_errs = self._calc_random_errs()
        if rand_errs is not None:
            measurements = measurements + rand_errs

        dep_sys_errs = self._calc_systematic_errs_dependent(measurements)
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
        pv_data = pv.PolyData(self.sensor_data.positions)
        return pv_data

