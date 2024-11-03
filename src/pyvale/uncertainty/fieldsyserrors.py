'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import copy
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.physics.field import IField
from pyvale.physics.fieldsampler import sample_field_with_sensor_data
from pyvale.sensors.sensordata import SensorData
from pyvale.numerical.spatialintegrator import ESpatialIntType
from pyvale.uncertainty.errorcalculator import (IErrCalculator,
                                                EErrType,
                                                EErrDependence)
from pyvale.uncertainty.driftcalculator import IDriftCalculator
from pyvale.uncertainty.randomgenerator import IGeneratorRandom


class SysErrPositionRand(IErrCalculator):
    __slots__ = ("_field","_sensor_data_nominal","_sensor_data_perturbed",
                 "_err_dep","_rand_err_xyz")

    def __init__(self,
                 field: IField,
                 sensor_data_nominal: SensorData,
                 rand_pos_xyz: tuple[IGeneratorRandom | None,
                                     IGeneratorRandom | None,
                                     IGeneratorRandom | None],
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT
                 ) -> None:

        self._field = field
        self._sensor_data_nominal = copy.deepcopy(sensor_data_nominal)
        self._sensor_data_perturbed = copy.deepcopy(sensor_data_nominal)
        self._rand_err_xyz = rand_pos_xyz
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_sensor_data_perturbed(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:

        self._sensor_data_perturbed.positions = perturb_sensor_positions(
            self._sensor_data_nominal.positions,
            None,
            self._rand_err_xyz,
        )

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)


class SysErrSpatialAverage(IErrCalculator):
    __slots__ = ("_field","_sensor_data_nominal","_sensor_data_perturbed",
                "_err_dep")

    def __init__(self,
                 field: IField,
                 sensor_data_nominal: SensorData,
                 spatial_averager: ESpatialIntType,
                 spatial_dims: np.ndarray,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT
                 ) -> None:

        self._field = field
        self._sensor_data_nominal = sensor_data_nominal
        self._sensor_data_perturbed = copy.deepcopy(sensor_data_nominal)
        self._sensor_data_perturbed.spatial_averager = spatial_averager
        self._sensor_data_perturbed.spatial_dims = spatial_dims
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_sensor_data_perturbed(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:
        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)


class SysErrSpatialAveragePosRand(IErrCalculator):
    __slots__ = ("_field","_sensor_data_nominal","_sensor_data_perturbed",
                "_err_dep","_rand_pos_xyz")

    def __init__(self,
                 field: IField,
                 sensor_data_nominal: SensorData,
                 spatial_averager: ESpatialIntType,
                 spatial_dims: np.ndarray,
                 rand_pos_xyz: tuple[IGeneratorRandom | None,
                                     IGeneratorRandom | None,
                                     IGeneratorRandom | None],
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._field = field
        self._sensor_data_nominal = sensor_data_nominal
        self._sensor_data_perturbed = copy.deepcopy(sensor_data_nominal)
        self._sensor_data_perturbed.spatial_averager = spatial_averager
        self._sensor_data_perturbed.spatial_dims = spatial_dims
        self._rand_pos_xyz = rand_pos_xyz
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_sensor_data_perturbed(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:

        self._sensor_data_perturbed.positions = perturb_sensor_positions(
            self._sensor_data_nominal.positions,
            None,
            self._rand_pos_xyz,
        )

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)


class SysErrTimeRand(IErrCalculator):
    __slots__ = ("_field","_sensor_data_nominal","_sensor_data_perturbed",
                "_err_dep","_rand_time")

    def __init__(self,
                field: IField,
                sensor_data_nominal: SensorData,
                rand_time: IGeneratorRandom,
                err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._field = field
        self._sensor_data_nominal = sensor_data_nominal
        self._sensor_data_perturbed = copy.deepcopy(sensor_data_nominal)
        self._rand_time = rand_time
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_sensor_data_perturbed(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:
        time_nominal = self._sensor_data_nominal.sample_times

        self._sensor_data_perturbed.sample_times = \
            time_nominal + \
            self._rand_time.generate(size=time_nominal.shape)

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)



class SysErrTimeDrift(IErrCalculator):
    __slots__ = ("_field","_sensor_data_nominal","_sensor_data_perturbed",
                "_err_dep","_drift_calc")

    def __init__(self,
                 field: IField,
                 sensor_data_nominal: SensorData,
                 drift_calc: IDriftCalculator,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._field = field
        self._sensor_data_nominal = sensor_data_nominal
        self._sensor_data_perturbed = copy.deepcopy(sensor_data_nominal)
        self._drift_calc = drift_calc
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_sensor_data_perturbed(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:
        time_nominal = self._sensor_data_nominal.sample_times

        self._sensor_data_perturbed.sample_times = \
            time_nominal + \
            self._drift_calc.calc_drift(time_nominal)

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)


class SysErrAngleOffset(IErrCalculator):
    __slots__ = ("_field","_sensor_data_nominal","_sensor_data_perturbed",
                "_err_dep","_offset_ang_zyx")

    def __init__(self,
                 field: IField,
                 sensor_data_nominal: SensorData,
                 offset_ang_zyx: np.ndarray,
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._field = field
        self._sensor_data_nominal = sensor_data_nominal
        self._sensor_data_perturbed = copy.deepcopy(sensor_data_nominal)
        self._offset_ang_zyx = offset_ang_zyx
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_sensor_data_perturbed(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:

        angles_perturbed = [None]*len(self._sensor_data_nominal.angles)
        for ii,rot_nom in enumerate(self._sensor_data_nominal.angles):
            rot = Rotation.from_euler("zyx",self._offset_ang_zyx,degrees=True)
            angles_perturbed[ii] = rot*rot_nom

        self._sensor_data_perturbed.angles = tuple(angles_perturbed)

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)



class SysErrAngleRand(IErrCalculator):
    __slots__ = ("_field","_sensor_data_nominal","_sensor_data_perturbed",
                "_err_dep","_rand_ang_zyx")

    def __init__(self,
                 field: IField,
                 sensor_data_nominal: SensorData,
                 rand_ang_zyx: tuple[IGeneratorRandom | None,
                                     IGeneratorRandom | None,
                                     IGeneratorRandom | None],
                 err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._field = field
        self._sensor_data_nominal = sensor_data_nominal


        self._sensor_data_perturbed = copy.deepcopy(sensor_data_nominal)
        if self._sensor_data_perturbed.angles is None:
            self._sensor_data_perturbed.angles = [
                Rotation.from_euler("zyx",[0,0,0], degrees=True)
                for _ in range(sensor_data_nominal.positions.shape[0])]

        self._rand_ang_zyx = rand_ang_zyx
        self._err_dep = err_dep

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_sensor_data_perturbed(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:

        # NOTE: lots of for loops here, can probably fix with matrices
        angles_perturbed = [None]*len(self._sensor_data_nominal.angles)
        for ii,rot_orig in enumerate(self._sensor_data_nominal.angles):

            rot_rand_angs = np.zeros((3,))
            for jj,rand_ang in enumerate(self._rand_ang_zyx):
                if rand_ang is not None:
                    rot_rand_angs[jj] = rand_ang.generate(size=1)

            rand_rot = Rotation.from_euler("zyx", rot_rand_angs, degrees=True)
            angles_perturbed[ii] = rand_rot*rot_orig

        self._sensor_data_perturbed.angles = tuple(angles_perturbed)

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)



@dataclass(slots=True)
class FieldErrorData:
    pos_offset_xyz: np.ndarray | None = None #shape=(n_sens,3 as {x,y,z})
    ang_offset_zyx: np.ndarray | None = None #shape=(n_sens,3 as {z,y,x})
    time_offset: np.ndarray | None = None #shape=(n_time_steps,)

    rand_pos_xyz: tuple[IGeneratorRandom | None,
                        IGeneratorRandom | None,
                        IGeneratorRandom | None] = (None,None,None)
    rand_ang_zyx: tuple[IGeneratorRandom | None,
                        IGeneratorRandom | None,
                        IGeneratorRandom | None] = (None,None,None)
    rand_time: IGeneratorRandom | None = None

    spatial_averager: ESpatialIntType | None = None
    spatial_dims: np.ndarray | None = None


class SysErrField(IErrCalculator):
    __slots__ = ("_field","_sensor_data_nominal","_sensor_data_perturbed",
                 "_field_err_data","_err_dep")

    def __init__(self,
                field: IField,
                sensor_data: SensorData,
                field_err_data: FieldErrorData,
                err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._field = field
        self._sensor_data_nominal = sensor_data
        self._sensor_data_perturbed = init_sensor_data_perturbed(sensor_data,
                                                                 field)
        self._field_err_data = field_err_data
        self._err_dep = err_dep

        self._sensor_data_perturbed.spatial_averager = \
            field_err_data.spatial_averager
        self._sensor_data_perturbed.spatial_dims = \
            field_err_data.spatial_dims

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_perturbed_sensor_data(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,err_basis: np.ndarray) -> tuple[np.ndarray,
                                                       SensorData | None]:

        self._sensor_data_perturbed.positions = perturb_sensor_positions(
            self._sensor_data_perturbed.positions,
            self._field_err_data.pos_offset_xyz,
            self._field_err_data.rand_pos_xyz,
        )

        self._sensor_data_perturbed.sample_times = perturb_sample_times(
            self._sensor_data_perturbed.sample_times,
            self._field_err_data.time_offset,
            self._field_err_data.rand_time,
            None,
        )

        self._sensor_data_perturbed.angles = perturb_sensor_angles(
            self._sensor_data_perturbed.angles,
            self._field_err_data.ang_offset_zyx,
            self._field_err_data.rand_ang_zyx,
        )

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)


#-------------------------------------------------------------------------------
def init_sensor_data_perturbed(sensor_data_nominal: SensorData,
                               field: IField) -> SensorData:
    sensor_data_perturbed = copy.deepcopy(sensor_data_nominal)

    if sensor_data_perturbed.sample_times is None:
        sensor_data_perturbed.sample_times = field.get_time_steps()

    if sensor_data_perturbed.angles is None:
        sensor_angles = [Rotation.from_euler("zyx",[0,0,0], degrees=True)
                         for _ in range(sensor_data_nominal.positions.shape[0])]

        sensor_data_perturbed.angles = tuple(sensor_angles)

    return sensor_data_perturbed


def perturb_sensor_positions(sens_pos_nominal: np.ndarray,
                             offset_pos_xyz: np.ndarray | None,
                             rand_pos_xyz: tuple[IGeneratorRandom | None,
                                                 IGeneratorRandom | None,
                                                 IGeneratorRandom | None],

                            ) -> np.ndarray:
    # NOTE: assumes all sensors have the same offset
    sens_pos_perturbed = np.copy(sens_pos_nominal)

    for ii,rng in enumerate(rand_pos_xyz):
        if offset_pos_xyz is not None:
            sens_pos_perturbed[:,ii] = sens_pos_perturbed[:,ii] + \
                offset_pos_xyz[:,ii]
        if rng is not None:
            sens_pos_perturbed[:,ii] = sens_pos_perturbed[:,ii] + \
                rng.generate(size=sens_pos_perturbed.shape[0])

    return sens_pos_perturbed

def perturb_sample_times(time_nominal: np.ndarray,
                         time_offset: float | None,
                         time_rand: IGeneratorRandom | None,
                         time_drift: IDriftCalculator | None
                         ) -> np.ndarray | None:

    if time_offset is None and time_rand is None and time_drift is None:
        return None

    time_perturbed = np.copy(time_nominal)

    if time_offset is not None:
        time_perturbed = time_perturbed + time_offset
    if time_rand is not None:
        time_perturbed = time_perturbed + time_rand.generate(
            size=time_nominal.shape)
    if time_drift is not None:
        time_perturbed = time_perturbed + time_drift.calc_drift(time_nominal)

    return time_perturbed


def perturb_sensor_angles(angles_nominal: tuple[Rotation,...],
                          angle_offsets_zyx: np.ndarray | None,
                          rand_ang_zyx: tuple[IGeneratorRandom | None,
                                              IGeneratorRandom | None,
                                              IGeneratorRandom | None] | None,
                          ) -> tuple[Rotation,...] | None:

    if angle_offsets_zyx is None and rand_ang_zyx is None:
        return None

    angles_perturbed = [None]*len(angles_nominal)
    for ii,rot_nom in enumerate(angles_nominal): # loop over sensors
        # NOTE: adding angles here might not be quite correct
        sensor_rot_angs = np.zeros((3,))

        if angle_offsets_zyx is not None:
            sensor_rot_angs = sensor_rot_angs + angle_offsets_zyx[ii,:]

        if rand_ang_zyx is not None:
            for jj,rand_ang in enumerate(rand_ang_zyx): # loop over components
                if rand_ang is not None:
                    sensor_rot_angs[jj] = sensor_rot_angs[jj] + \
                        rand_ang.generate(size=1)

        sensor_rot = Rotation.from_euler("zyx",sensor_rot_angs, degrees=True)
        angles_perturbed[ii] = sensor_rot*rot_nom

    return tuple(angles_perturbed)

