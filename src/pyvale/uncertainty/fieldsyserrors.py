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
from pyvale.numerical.spatialintegrator import ISpatialAverager
from pyvale.uncertainty.errorcalculator import (IErrCalculator,
                                                ErrorData,
                                                EErrorType,
                                                EErrorCalc)
from pyvale.uncertainty.driftcalculator import IDriftCalculator
from pyvale.uncertainty.randomgenerator import IGeneratorRandom


class SysErrRandPosition(IErrCalculator):
    def __init__(self,
                 field: IField,
                 sens_pos: np.ndarray,
                 rand_by_ax: tuple[IGeneratorRandom | None,
                                   IGeneratorRandom | None,
                                   IGeneratorRandom | None],
                 sample_times: np.ndarray | None = None,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:

        self._field = field
        self._sens_pos_original = np.copy(sens_pos)
        self._sens_pos_perturbed = np.copy(sens_pos)
        self._rand_by_ax = rand_by_ax
        self._sample_times = sample_times
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

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
                 spatial_averager: ISpatialAverager,
                 sample_times: np.ndarray | None = None,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:

        self._field = field
        self._spatial_averager = spatial_averager
        self._sample_times = sample_times
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        sys_errs = self._spatial_averager.get_averages() - err_basis

        err_data = ErrorData(error_array=sys_errs)
        return err_data


class SysErrSpatialAverageRandPos(IErrCalculator):
    def __init__(self,
                 field: IField,
                 spatial_average: ISpatialAverager,
                 sens_pos: np.ndarray,
                 rand_by_ax: tuple[IGeneratorRandom | None,
                                   IGeneratorRandom | None,
                                   IGeneratorRandom | None],
                 sample_times: np.ndarray | None = None,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:

        self._field = field
        self._spatial_average = spatial_average
        self._sens_pos_original = np.copy(sens_pos)
        self._sens_pos_perturbed = np.copy(sens_pos)
        self._rand_by_ax = rand_by_ax
        self._sample_times = sample_times
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

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
                rand_time: IGeneratorRandom,
                sample_times: np.ndarray | None = None,
                err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._rand_time = rand_time
        self._err_calc = err_calc

        if sample_times is None:
            original_times = field.get_time_steps()
        else:
            original_times = sample_times

        self._time_original = np.copy(original_times)
        self._time_perturbed = np.copy(original_times)

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

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
                 sample_times: np.ndarray | None = None,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._drift = drift
        self._err_calc = err_calc

        if sample_times is None:
            original_times = field.get_time_steps()
        else:
            original_times = sample_times

        self._time_original = np.copy(original_times)
        self._time_perturbed = np.copy(original_times)

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

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


class SysErrAngleOffset(IErrCalculator):
    def __init__(self,
                 field: IField,
                 sens_pos: np.ndarray,
                 angles: tuple[Rotation,...],
                 offset_ang_zyx: np.ndarray,
                 sample_times: np.ndarray | None = None,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._sens_angles_original = angles
        self._sens_angles_perturbed = angles
        self._offset_ang_zyx = offset_ang_zyx
        self._sample_times = sample_times
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def get_perturbed_angles(self) -> tuple[Rotation,...]:
        return self._sens_angles_perturbed

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        # NOTE: lots of for loops here, can probably fix with matrices
        self._sens_angles_perturbed = [None]*len(self._sens_angles_original)
        for ii,rot_orig in enumerate(self._sens_angles_original):
            rot = Rotation.from_euler("zyx",
                                           self._offset_ang_zyx,
                                           degrees=True)
            self._sens_angles_perturbed[ii] = rot*rot_orig

        self._sens_angles_perturbed = tuple(self._sens_angles_perturbed)
        sys_errs = self._field.sample_field(self._sens_pos,
                                            self._sample_times,
                                            self._sens_angles_perturbed) \
                                            - err_basis

        err_data = ErrorData(error_array=sys_errs,
                             angles=self._sens_angles_perturbed)
        return err_data


class SysErrAngleRand(IErrCalculator):
    def __init__(self,
                 field: IField,
                 sens_pos: np.ndarray,
                 angles: tuple[Rotation,...],
                 rand_ang_zyx: tuple[IGeneratorRandom | None,
                                    IGeneratorRandom | None,
                                    IGeneratorRandom | None],
                 sample_times: np.ndarray | None = None,
                 err_calc: EErrorCalc = EErrorCalc.INDEPENDENT) -> None:

        self._field = field
        self._sens_pos = sens_pos
        self._sens_angles_original = angles
        self._sens_angles_perturbed = angles
        self._rand_ang_zyx = rand_ang_zyx
        self._sample_times = sample_times
        self._err_calc = err_calc

    def get_error_calc(self) -> EErrorCalc:
        return self._err_calc

    def get_error_type(self) -> EErrorType:
        return EErrorType.SYSTEMATIC

    def get_perturbed_angles(self) -> tuple[Rotation,...]:
        return self._sens_angles_perturbed

    def calc_errs(self, err_basis: np.ndarray) -> ErrorData:

        # NOTE: lots of for loops here, can probably fix with matrices
        self._sens_angles_perturbed = [None]*len(self._sens_angles_original)
        for ii,rot_orig in enumerate(self._sens_angles_original):

            rot_rand_list = np.zeros((3,))
            for jj,rand_ang in enumerate(self._rand_ang_zyx):
                if rand_ang is not None:
                    rot_rand_list[jj] = rand_ang.generate(size=1)

            rand_rot = Rotation.from_euler("zyx", rot_rand_list, degrees=True)
            self._sens_angles_perturbed[ii] = rand_rot*rot_orig

        self._sens_angles_perturbed = tuple(self._sens_angles_perturbed)
        sys_errs = self._field.sample_field(self._sens_pos,
                                            self._sample_times,
                                            self._sens_angles_perturbed) \
                                            - err_basis

        err_data = ErrorData(error_array=sys_errs,
                             angles=self._sens_angles_perturbed)
        return err_data
