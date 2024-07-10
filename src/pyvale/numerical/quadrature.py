'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

from pyvale.physics.field import IField
from pyvale.numerical.spatialintegrator import ISpatialIntegrator


def create_gauss_pt_array(gauss_pt_offsets: np.ndarray,
                        cent_pos: np.ndarray,
                        ) -> np.ndarray:
    offset_array = np.tile(gauss_pt_offsets,(cent_pos.shape[0],1))
    gauss_pt_array = np.repeat(cent_pos,gauss_pt_offsets.shape[0],axis=0)
    return gauss_pt_array + offset_array


class Quad2D(ISpatialIntegrator):
    def __init__(self,
                 gauss_pt_offsets: np.ndarray,
                 field: IField,
                 cent_pos: np.ndarray,
                 dims: np.ndarray,
                 sample_times: np.ndarray | None = None) -> None:

        self._field = field
        self._cent_pos = cent_pos
        self._dims = dims
        self._sample_times = sample_times

        self._area = self._dims[0]*self._dims[1]

        self._n_gauss_pts = 4
        self._gauss_pt_offsets = self._dims * 1/np.sqrt(3)* np.array([[-1,-1,0],
                                                            [-1,1,0],
                                                            [1,-1,0],
                                                            [1,1,0]])

        self._gauss_pts = create_gauss_pt_array(self._gauss_pt_offsets,
                                                cent_pos)
        self._integrals = self.calc_integrals(None, sample_times)

    def calc_integrals(self,
                       cent_pos: np.ndarray | None = None,
                       sample_times: np.ndarray | None = None) -> np.ndarray:

        if cent_pos is not None:
            self._gauss_pts = create_gauss_pt_array(self._gauss_pt_offsets,
                                                    cent_pos)

        # Shape=(n_gauss_pts*n_sens,n_comps,n_timesteps)
        gauss_vals = self._field.sample_field(self._gauss_pts,
                                            sample_times)
        # Shape=(n_gauss_pts,n_sens,n_comps,n_timesteps)
        gauss_vals = gauss_vals.reshape((self._n_gauss_pts,
                                         self._cent_pos.shape[0],
                                         gauss_vals.shape[1],
                                         gauss_vals.shape[2]),
                                         order='F')

        # Shape=(n_sensors,n_comps,n_timsteps)
        # NOTE: coeff comes from changing gauss interval from [-1,1] to [a,b] -
        # so (a-b)/2 * (a-b)/2 = sensor_area / 4, then need to divide by the
        # integration area to convert to an average.
        self._integrals = self._area/4 * np.sum(gauss_vals,axis=0)

        return self._integrals

    def get_integrals(self) -> np.ndarray:
        return self._integrals

    def calc_averages(self,
                      cent_pos: np.ndarray | None = None,
                      sample_times: np.ndarray | None = None) -> np.ndarray:
        return (1/self._area)*self.calc_integrals(cent_pos,sample_times)

    def get_averages(self) -> np.ndarray:
        return (1/self._area)*self._integrals


class Disc2D:
    def __init__(self,
                 pos: np.ndarray,
                 rad: float) -> None:
        self._pos = pos
        self._rad = rad


class QuadratureFactory:
    @staticmethod
    def quad_2d_4points(field: IField,
                 cent_pos: np.ndarray,
                 dims: np.ndarray,
                 sample_times: np.ndarray | None = None) -> Quad2D:

        gauss_pt_offsets = dims * 1/np.sqrt(3)* np.array([[-1,-1,0],
                                                        [-1,1,0],
                                                        [1,-1,0],
                                                        [1,1,0]])
        quadrature = Quad2D(gauss_pt_offsets,field,cent_pos,dims,sample_times)
        return quadrature



