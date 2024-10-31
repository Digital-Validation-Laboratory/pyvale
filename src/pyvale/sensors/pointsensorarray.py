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


@dataclass(slots=True)
class SensorData:
    positions: np.ndarray | None = None
    sample_times: np.ndarray | None = None
    area_averager: ISpatialAverager | None = None
    angles: tuple[Rotation,...] | None = None


class PointSensorArray:
    __slots__ = ("sensor_data","field","descriptor","_truth","_measurements",
                 "_error_integrator")

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
        self._error_integrator = None

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
    # Truth calculation from simulation
    def calc_truth_values(self) -> np.ndarray:
        if self.sensor_data.area_averager is None:
            return self.field.sample_field(self.sensor_data.positions,
                                           self.sensor_data.sample_times,
                                           self.sensor_data.angles)

        return self.sensor_data.area_averager.calc_averages(
                        self.sensor_data.positions,
                        self.sensor_data.sample_times)

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth_values()

        return self._truth

    #---------------------------------------------------------------------------
    # Errors
    def set_error_integrator(self, err_int: ErrorIntegrator) -> None:
        self._error_integrator = err_int

    def get_systematic_errors(self) -> np.ndarray:
        return self._error_integrator.get_errs_systematic()

    def get_random_errors(self) -> np.ndarray:
        return self._error_integrator.get_errs_random()

    def get_total_errors(self) -> np.ndarray:
        return self._error_integrator.get_errs_total()

    #---------------------------------------------------------------------------
    # Measurements
    def calc_measurements(self) -> np.ndarray:
        if self._error_integrator is None:
            self._measurements = self.get_truth()
        else:
            self._measurements = self.get_truth() + \
                self._error_integrator.calc_errors_from_chain(self.get_truth())

        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.calc_measurements()

        return self._measurements

    #---------------------------------------------------------------------------
    # Visualisation tools
    # TODO: this should probably be moved from here
    def get_visualiser(self) -> pv.PolyData:
        return pv.PolyData(self.sensor_data.positions)
