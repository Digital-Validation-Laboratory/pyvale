'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''

import numpy as np

from pyvale.physics.field import IField
from pyvale.numerical.spatialintegrator import ISpatialIntegrator
from pyvale.uncertainty.errorcalculator import IErrCalculator
from pyvale.uncertainty.driftcalculator import IDriftCalculator


class SysErrRandPosition(IErrCalculator):

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
                 spatial_average: ISpatialIntegrator,
                 sample_times: np.ndarray | None = None,
                 ) -> None:

        self._field = field
        self._spatial_average = spatial_average
        self._sample_times = sample_times

    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        sys_errs = self._spatial_average.get_averages() - err_basis
        return sys_errs


class SysErrSpatialAverageRandPos(IErrCalculator):

    def __init__(self,
                 field: IField,
                 spatial_average: ISpatialIntegrator,
                 sens_pos: np.ndarray,
                 std_by_ax: tuple[float | None,float | None,float | None],
                 sample_times: np.ndarray | None = None,
                 seed: int | None = None) -> None:

        self._field = field
        self._spatial_average = spatial_average
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

        sys_errs = self._spatial_average.calc_averages(self._sens_pos_perturbed,
                                            self._sample_times) - err_basis
        return sys_errs


class SysErrTimeRand(IErrCalculator):

    def __init__(self,
                field: IField,
                sens_pos: np.ndarray,
                time_std: float,
                sample_times: np.ndarray | None = None,
                seed: int | None = None) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._time_std = time_std

        if sample_times is None:
            original_times = field.get_time_steps()
        else:
            original_times = sample_times

        self._time_original = np.copy(original_times)
        self._time_perturbed = np.copy(original_times)

        self._rng = np.random.default_rng(seed)

    def get_perturbed_time(self) -> np.ndarray:
        return self._time_perturbed

    def calc_errs(self,
                err_basis: np.ndarray) -> np.ndarray:

        self._time_perturbed = self._time_original + \
                                self._rng.normal(loc=0.0,
                                                scale=self._time_std,
                                                size=self._time_original.shape)

        sys_errs = self._field.sample_field(self._sens_pos,
                            self._time_perturbed) - err_basis

        return sys_errs



class SysErrTimeDrift(IErrCalculator):

    def __init__(self,
                field: IField,
                sens_pos: np.ndarray,
                drift: IDriftCalculator,
                sample_times: np.ndarray | None = None) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._drift = drift

        if sample_times is None:
            original_times = field.get_time_steps()
        else:
            original_times = sample_times

        self._time_original = np.copy(original_times)
        self._time_perturbed = np.copy(original_times)


    def get_perturbed_time(self) -> np.ndarray:
        return self._time_perturbed

    def calc_errs(self,
                  err_basis: np.ndarray) -> np.ndarray:

        self._time_perturbed = self._time_original + \
            self._drift.calc_drift(self._time_original)

        print(80*"=")
        print(f"{self._time_perturbed=}")
        print(80*"=")

        sys_errs = self._field.sample_field(self._sens_pos,
                                    self._time_perturbed) - err_basis

        return sys_errs
