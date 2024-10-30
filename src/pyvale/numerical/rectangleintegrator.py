'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np

from pyvale.physics.field import IField
from pyvale.numerical.spatialintegrator import (ISpatialIntegrator,
                                                create_int_pt_array)

# NOTE: code below is very similar to quadrature integrator should be able to
# refactor into injected classes/functions
class Rectangle2D(ISpatialIntegrator):
    def __init__(self,
                 int_pt_offsets: np.ndarray,
                 field: IField,
                 cent_pos: np.ndarray,
                 dims: np.ndarray,
                 sample_times: np.ndarray | None = None) -> None:

        self._field = field
        self._cent_pos = cent_pos
        self._dims = dims
        self._sample_times = sample_times

        # TODO: check that this works for non-square averages
        self._area_tot = self._dims[0]*self._dims[1]
        self._area_int = self._area_tot/int_pt_offsets.shape[0]

        self._n_int_pts = int_pt_offsets.shape[0]
        self._int_pt_offsets = int_pt_offsets
        self._int_pts = create_int_pt_array(self._int_pt_offsets,
                                            cent_pos)

        self._integrals = self.calc_integrals(None, sample_times)


    def calc_integrals(self,
                    cent_pos: np.ndarray | None = None,
                    sample_times: np.ndarray | None = None) -> np.ndarray:

        if cent_pos is not None:
            # shape=(n_sens*n_gauss_pts,n_dims)
            self._gauss_pts = create_int_pt_array(self._int_pt_offsets,
                                                  cent_pos)

        # shape=(n_gauss_pts*n_sens,n_comps,n_timesteps)
        int_vals = self._field.sample_field(self._int_pts,
                                              sample_times)

        meas_shape = (self._cent_pos.shape[0],
                int_vals.shape[1],
                int_vals.shape[2])

        # shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        int_vals = int_vals.reshape((self._n_int_pts,)+meas_shape,
                                         order='F')

        # shape=(n_sensors,n_comps,n_timsteps)
        self._integrals = np.sum(self._area_int*int_vals,axis=0)

        return self._integrals

    def get_integrals(self) -> np.ndarray:
        return self._integrals

    def calc_averages(self,
                      cent_pos: np.ndarray | None = None,
                      sample_times: np.ndarray | None = None) -> np.ndarray:
        return (1/self._area_tot)*self.calc_integrals(cent_pos,sample_times)

    def get_averages(self) -> np.ndarray:
        return (1/self._area_tot)*self._integrals


