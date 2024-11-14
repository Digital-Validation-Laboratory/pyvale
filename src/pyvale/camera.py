'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
from pyvale.field import IField
from pyvale.sensor import ISensor
from pyvale.errorintegrator import ErrIntegrator
from pyvale.sensordescriptor import SensorDescriptor
from pyvale.cameradata import CameraData
from pyvale.fieldsampler import sample_field_with_camera_data


class CameraBasic2D(ISensor):
    __slots__ = ("_cam_data",)

    def __init__(self,
                 cam_data: CameraData,
                 field: IField,
                 descriptor: SensorDescriptor | None = None,
                 ) -> None:
        self.cam_data = cam_data
        self.field = field
        self.error_integrator = None

        self.descriptor = SensorDescriptor()
        if descriptor is not None:
            self.descriptor = descriptor

        self._truth = None
        self._measurements = None

    #---------------------------------------------------------------------------
    # Truth calculation from simulation
    def calc_truth_values(self) -> np.ndarray:
        self._truth = sample_field_with_camera_data(self.field,
                                                    self.cam_data)
        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth_values()
        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self._truth

    #---------------------------------------------------------------------------
    # Errors
    def set_error_integrator(self, err_int: ErrIntegrator) -> None:
        self.error_integrator = err_int

    def get_errors_systematic(self) -> np.ndarray | None:
        if self.error_integrator is None:
            return None

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self.error_integrator.get_errs_systematic()

    def get_errors_random(self) -> np.ndarray | None:
        if self.error_integrator is None:
            return None

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self.error_integrator.get_errs_random()

    def get_errors_total(self) -> np.ndarray | None:
        if self.error_integrator is None:
            return None

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self.error_integrator.get_errs_total()

    #---------------------------------------------------------------------------
    # Measurements
    def calc_measurements(self) -> np.ndarray:
        if self.error_integrator is None:
            self._measurements = self.get_truth()
        else:
            self._measurements = self.get_truth() + \
                self.error_integrator.calc_errors_from_chain(self.get_truth())

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.calc_measurements()

        #shape=(n_sensors,n_field_comps,n_time_steps)
        return self._measurements

