'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''

import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.physics.field import IField
from pyvale.numerical.spatialintegrator import ISpatialIntegrator
from pyvale.uncertainty.errorcalculator import IErrCalculator, ErrorData
from pyvale.uncertainty.driftcalculator import IDriftCalculator
from pyvale.uncertainty.randomgenerator import IRandomGenerator


class SysErrRandPosition(IErrCalculator):

    def __init__(self,
                 field: IField,
                 sens_pos: np.ndarray,
                 rand_by_ax: tuple[IRandomGenerator | None,
                                   IRandomGenerator | None,
                                   IRandomGenerator | None],
                 sample_times: np.ndarray | None = None) -> None:

        self._field = field
        self._sens_pos_original = np.copy(sens_pos)
        self._sens_pos_perturbed = np.copy(sens_pos)
        self._rand_by_ax = rand_by_ax
        self._sample_times = sample_times

    def get_perturbed_pos(self) -> np.ndarray:
        return self._sens_pos_perturbed

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        self._sens_pos_perturbed = np.copy(self._sens_pos_original)

        for ii,rng in enumerate(self._rand_by_ax):
            if rng is not None:
                self._sens_pos_perturbed[:,ii] = self._sens_pos_perturbed[:,ii]\
                    + rng.generate(size=self._sens_pos_perturbed.shape[0])

        sys_errs = self._field.sample_field(self._sens_pos_perturbed,
                                            self._sample_times) - err_basis

        err_data = ErrorData(error_array=sys_errs,
                             positions=self._sens_pos_perturbed)
        return err_data


class SysErrSpatialAverage(IErrCalculator):

    def __init__(self,
                 field: IField,
                 spatial_averager: ISpatialIntegrator,
                 sample_times: np.ndarray | None = None,
                 ) -> None:

        self._field = field
        self._spatial_averager = spatial_averager
        self._sample_times = sample_times

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        sys_errs = self._spatial_averager.get_averages() - err_basis

        err_data = ErrorData(error_array=sys_errs)
        return err_data


class SysErrSpatialAverageRandPos(IErrCalculator):

    def __init__(self,
                 field: IField,
                 spatial_average: ISpatialIntegrator,
                 sens_pos: np.ndarray,
                 rand_by_ax: tuple[IRandomGenerator | None,
                                   IRandomGenerator | None,
                                   IRandomGenerator | None],
                 sample_times: np.ndarray | None = None) -> None:

        self._field = field
        self._spatial_average = spatial_average
        self._sens_pos_original = np.copy(sens_pos)
        self._sens_pos_perturbed = np.copy(sens_pos)
        self._rand_by_ax = rand_by_ax
        self._sample_times = sample_times

    def get_perturbed_pos(self) -> np.ndarray:
        return self._sens_pos_perturbed

    def calc_errs(self, err_basis: np.ndarray) -> np.ndarray:

        self._sens_pos_perturbed = np.copy(self._sens_pos_original)

        for ii,rng in enumerate(self._rand_by_ax):
            if rng is not None:
                self._sens_pos_perturbed[:,ii] = self._sens_pos_perturbed[:,ii]\
                    + rng.generate(size=self._sens_pos_perturbed.shape[0])

        sys_errs = self._spatial_average.calc_averages(self._sens_pos_perturbed,
                                            self._sample_times) - err_basis

        err_data = ErrorData(error_array=sys_errs,
                             positions=self._sens_pos_perturbed)
        return err_data


class SysErrTimeRand(IErrCalculator):

    def __init__(self,
                field: IField,
                sens_pos: np.ndarray,
                rand_time: IRandomGenerator,
                sample_times: np.ndarray | None = None) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._rand_time = rand_time

        if sample_times is None:
            original_times = field.get_time_steps()
        else:
            original_times = sample_times

        self._time_original = np.copy(original_times)
        self._time_perturbed = np.copy(original_times)


    def get_perturbed_time(self) -> np.ndarray:
        return self._time_perturbed

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        self._time_perturbed = self._time_original + \
                                self._rand_time.generate(
                                    size=self._time_original.shape)

        sys_errs = self._field.sample_field(self._sens_pos,
                            self._time_perturbed) - err_basis

        err_data = ErrorData(error_array=sys_errs,
                             time_steps=self._time_perturbed)
        return err_data


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

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        self._time_perturbed = self._time_original + \
            self._drift.calc_drift(self._time_original)

        sys_errs = self._field.sample_field(self._sens_pos,
                                    self._time_perturbed) - err_basis

        err_data = ErrorData(error_array=sys_errs,
                             time_steps=self._time_perturbed)
        return err_data


class SysErrOrientation(IErrCalculator):

    def __init__(self,
                 field: IField,
                 sens_pos: np.ndarray,
                 angles: tuple[Rotation,...],
                 rand_by_ax: tuple[IRandomGenerator | None,
                                   IRandomGenerator | None,
                                   IRandomGenerator | None],
                 sample_times: np.ndarray | None = None) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._sens_angles_original = angles
        self._sens_angles_perturbed = angles
        self._rand_by_ax = rand_by_ax
        self._sample_times = sample_times

    def get_perturbed_angles(self) -> tuple[Rotation,...]:
        return self._sens_angles_perturbed

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        self._sens_angles_perturbed = self._sens_angles_original

        for ii,rng in enumerate(self._rand_by_ax):
            if rng is not None:
                pass

        sys_errs = self._field.sample_field(self._sens_pos,
                                            self._sample_times,
                                            self._sens_angles_perturbed) \
                                            - err_basis

        err_data = ErrorData(error_array=sys_errs,
                             positions=self._sens_pos_perturbed)
        return err_data
