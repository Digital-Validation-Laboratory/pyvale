'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

import numpy as np
import pyvista as pv

from pyvale.field import Field
from pyvale.sensorarray import SensorArray, MeasurementData
from pyvale.uncertainty.syserrintegrator import SysErrIntegrator
from pyvale.uncertainty.randerrintegrator import RandErrIntegrator


class PointSensorArray(SensorArray):
    def __init__(self,
                 positions: np.ndarray,
                 field: Field,
                 sample_times: np.ndarray | None = None
                 ) -> None:

        self._positions = positions
        self._field = field
        self._sample_times = sample_times

        self._sys_err_int = None
        self._rand_err_int = None


    def get_field(self) -> Field:
        return self._field

    def get_positions(self) -> np.ndarray:
        return self._positions

    def get_sample_times(self) -> np.ndarray:
        if self._sample_times is None:
            return self._field.get_time_steps()

        return self._sample_times

    def get_num_sensors(self) -> int:
        return self._positions.shape[0]

    def get_measurement_shape(self) -> tuple[int,int,int]:
        return (self.get_num_sensors(),
                len(self._field.get_all_components()),
                self.get_sample_times().shape[0])


    def get_truth_values(self) -> np.ndarray:
        return self._field.sample_field(self._positions,
                                        self._sample_times)


    def set_sys_err_integrator(self,
                               err_int: SysErrIntegrator) -> None:
        self._sys_err_int = err_int


    def get_systematic_errs(self) -> np.ndarray | None:
        if self._sys_err_int is None:
            return None

        return self._sys_err_int.get_sys_errs_tot()


    def set_rand_err_integrator(self,
                                err_int: RandErrIntegrator) -> None:
        self._rand_err_int = err_int


    def get_random_errs(self) -> np.ndarray | None:
        if self._rand_err_int is None:
            return None

        return self._rand_err_int.get_rand_errs_tot()


    def get_measurements(self) -> np.ndarray:
        measurements = self.get_truth_values()
        sys_errs = self.get_systematic_errs()
        rand_errs = self.get_random_errs()

        if sys_errs is not None:
            measurements = measurements + sys_errs

        if rand_errs is not None:
            measurements = measurements + rand_errs

        return measurements


    def get_measurement_data(self) -> MeasurementData:
        measurement_data = MeasurementData()
        measurement_data.measurements = self.get_measurements()
        measurement_data.systematic_errs = self.get_systematic_errs()
        measurement_data.random_errs = self.get_random_errs()
        measurement_data.truth_values = self.get_truth_values()
        return measurement_data


    def get_visualiser(self) -> pv.PolyData:
        pv_data = pv.PolyData(self._positions)
        #pv_data['labels'] = self._sensor_names
        return pv_data

