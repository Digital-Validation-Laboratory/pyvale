'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

import numpy as np

from pyvale.physics.field import IField
from pyvale.uncertainty.errorcalculator import IErrCalculator


class SysErrPosition(IErrCalculator):

    def __init__(self,
                 field: IField,
                 sens_pos: np.ndarray,
                 std_by_ax: tuple[float | None,float | None,float | None],
                 sample_times: np.ndarray | None = None,
                 seed: int | None = None) -> None:

        self._field = field
        self._sens_pos_original = np.copy(sens_pos)
        self._sens_pos_perturbed = np.copy(sens_pos)
        self._std_by_ax = std_by_ax
        self._sample_times = sample_times
        self._rng = np.random.default_rng(seed)

    def get_perturbed_pos(self) -> np.ndarray:

        return self._sens_pos_perturbed

    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        self._sens_pos_perturbed = np.copy(self._sens_pos_original)

        for ii,ss in enumerate(self._std_by_ax):
            if ss is not None:
                self._sens_pos_perturbed[:,ii] = self._sens_pos_perturbed[:,ii]\
                    + self._rng.normal(loc=0.0,
                                        scale=ss,
                                        size=self._sens_pos_perturbed.shape[0])

        sys_errs = self._field.sample_field(self._sens_pos_perturbed,
                                            self._sample_times) - err_basis
        return sys_errs


class SysErrSpatialAverage(IErrCalculator):

    def __init__(self,
                 field: IField,
                 sens_pos: np.ndarray,
                 sens_dims: np.ndarray,
                 sample_times: np.ndarray | None = None,
                 ) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._sens_dims = sens_dims
        self._sens_area = self._sens_dims[0]*self._sens_dims[1]

        self._sample_times = sample_times

        self._n_gauss_pts = 4
        gauss_pts = self._sens_dims * 1/np.sqrt(3)* np.array([[-1,-1,0],
                                                            [-1,1,0],
                                                            [1,-1,0],
                                                            [1,1,0]])

        gauss_pos = np.repeat(self._sens_pos,self._n_gauss_pts,axis=0)
        gauss_pts = np.tile(gauss_pts,(sens_pos.shape[0],1))
        self._sens_gauss_pos = gauss_pts + gauss_pos

    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        gauss_vals = self._field.sample_field(self._sens_gauss_pos,
                                                self._sample_times)

        gauss_vals = gauss_vals.reshape((self._n_gauss_pts,
                                         self._sens_pos.shape[0],
                                         gauss_vals.shape[1],
                                         gauss_vals.shape[2]),
                                         order='F')

        # NOTE: coeff comes from changing gauss interval from [-1,1] to [a,b] -
        # so (a-b)/2 * (a-b)/2 = sensor_area / 4, then need to divide by the
        # sensor area to convert to an average. So, coeff=1/4.
        sens_vals = (1/4)*np.sum(gauss_vals,axis=0)

        sys_errs = sens_vals - err_basis
        return sys_errs


