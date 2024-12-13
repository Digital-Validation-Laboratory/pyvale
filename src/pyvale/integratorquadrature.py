'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Callable
import numpy as np
from pyvale.field import IField
from pyvale.integratorspatial import (IIntegratorSpatial,
                                     create_int_pt_array)
from pyvale.sensordata import SensorData


def create_gauss_weights_2d_4pts(meas_shape: tuple[int,int,int]) -> np.ndarray:
    return np.ones((4,)+meas_shape)


def create_gauss_weights_2d_9pts(meas_shape: tuple[int,int,int]) -> np.ndarray:
    # shape=(9,)+meas_shape
    gauss_weights = np.vstack((25/81 * np.ones((4,)+meas_shape),
                            40/81 * np.ones((4,)+meas_shape),
                            64/81 * np.ones((1,)+meas_shape)))
    return gauss_weights


class Quadrature2D(IIntegratorSpatial):
    __slots__ = ("_field","_area","_n_gauss_pts","_gauss_pt_offsets"
                 ,"_gauss_weight_func","_gauss_pts","_averages","_sens_data")

    def __init__(self,
                 field: IField,
                 sens_data: SensorData,
                 gauss_pt_offsets: np.ndarray,
                 gauss_weight_func: Callable) -> None:

        self._field = field
        self._sens_data = sens_data
        self._area = self._sens_data.spatial_dims[0] * \
            self._sens_data.spatial_dims[1]

        self._n_gauss_pts = gauss_pt_offsets.shape[0]
        self._gauss_pt_offsets = gauss_pt_offsets
        self._gauss_weight_func = gauss_weight_func

        self._gauss_pts = create_int_pt_array(self._sens_data,
                                              self._gauss_pt_offsets)
        self._averages = None

    def calc_integrals(self, sens_data: SensorData | None = None) -> np.ndarray:
        self._averages = self.calc_averages(sens_data)
        return self._area*self.get_averages()

    def get_integrals(self) -> np.ndarray:
        return self._area*self.get_averages()

    def calc_averages(self, sens_data: SensorData | None = None) -> np.ndarray:

        if sens_data is not None:
            self._sens_data = sens_data

        # shape=(n_sens*n_gauss_pts,n_dims)
        self._gauss_pts = create_int_pt_array(self._sens_data,
                                              self._gauss_pt_offsets)

        # shape=(n_gauss_pts*n_sens,n_comps,n_timesteps)
        gauss_vals = self._field.sample_field(self._gauss_pts,
                                              self._sens_data.sample_times,
                                              self._sens_data.angles)

        meas_shape = (self._sens_data.positions.shape[0],
                      gauss_vals.shape[1],
                      gauss_vals.shape[2])

        # shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        gauss_vals = gauss_vals.reshape((self._n_gauss_pts,)+meas_shape,
                                         order='F')

        # shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        gauss_weights = self._gauss_weight_func(meas_shape)

        # NOTE: coeff comes from changing gauss interval from [-1,1] to [a,b] -
        # so (a-b)/2 * (a-b)/2 = sensor_area / 4, then need to divide by the
        # integration area to convert to an average:
        # integrals = self._area/4 * np.sum(gauss_weights*gauss_vals,axis=0)
        # self._averages = (1/self._area)*integrals

        # shape=(n_sensors,n_comps,n_timsteps)=meas_shape
        self._averages = 1/4 * np.sum(gauss_weights*gauss_vals,axis=0)
        return self._averages

    def get_averages(self) -> np.ndarray:
        if self._averages is None:
            self._averages = self.calc_averages()

        return self._averages

