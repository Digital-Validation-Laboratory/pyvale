'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import copy
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.core.field import IField
from pyvale.core.fieldsampler import sample_field_with_sensor_data
from pyvale.core.sensordata import SensorData
from pyvale.core.integratortype import EIntSpatialType
from pyvale.core.errorcalculator import (IErrCalculator,
                                    EErrType,
                                    EErrDependence)
from pyvale.core.errordriftcalc import IDriftCalculator
from pyvale.core.generatorsrandom import IGeneratorRandom


@dataclass(slots=True)
class ErrFieldData:
    pos_offset_xyz: np.ndarray | None = None #shape=(n_sens,3 as {x,y,z})
    ang_offset_zyx: np.ndarray | None = None #shape=(n_sens,3 as {z,y,x})
    time_offset: np.ndarray | None = None #shape=(n_time_steps,)

    pos_rand_xyz: tuple[IGeneratorRandom | None,
                        IGeneratorRandom | None,
                        IGeneratorRandom | None] = (None,None,None)
    ang_rand_zyx: tuple[IGeneratorRandom | None,
                        IGeneratorRandom | None,
                        IGeneratorRandom | None] = (None,None,None)
    time_rand: IGeneratorRandom | None = None

    #TODO: implement drift for other dimensions, pos/angle
    time_drift: IDriftCalculator | None = None

    spatial_averager: EIntSpatialType | None = None
    spatial_dims: np.ndarray | None = None


class ErrSysField(IErrCalculator):
    __slots__ = ("_field","_sensor_data_perturbed","_field_err_data","_err_dep")

    def __init__(self,
                field: IField,
                field_err_data: ErrFieldData,
                err_dep: EErrDependence = EErrDependence.INDEPENDENT) -> None:

        self._field = field
        self._field_err_data = field_err_data
        self._err_dep = err_dep
        self._sensor_data_perturbed = SensorData()

    def get_error_dep(self) -> EErrDependence:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDependence) -> None:
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_perturbed_sensor_data(self) -> SensorData:
        return self._sensor_data_perturbed

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:

        self._sensor_data_perturbed = copy.deepcopy(sens_data)
        self._sensor_data_perturbed.spatial_averager = \
            self._field_err_data.spatial_averager
        self._sensor_data_perturbed.spatial_dims = \
            self._field_err_data.spatial_dims

        self._sensor_data_perturbed.positions = perturb_sensor_positions(
            self._sensor_data_perturbed.positions,
            self._field_err_data.pos_offset_xyz,
            self._field_err_data.pos_rand_xyz,
        )

        self._sensor_data_perturbed.sample_times = perturb_sample_times(
            self._field.get_time_steps(),
            self._sensor_data_perturbed.sample_times,
            self._field_err_data.time_offset,
            self._field_err_data.time_rand,
            self._field_err_data.time_drift,
        )

        self._sensor_data_perturbed.angles = perturb_sensor_angles(
            sens_data.positions.shape[0],
            self._sensor_data_perturbed.angles,
            self._field_err_data.ang_offset_zyx,
            self._field_err_data.ang_rand_zyx,
        )

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)




#-------------------------------------------------------------------------------
def perturb_sensor_positions(sens_pos_nominal: np.ndarray,
                             pos_offset_xyz: np.ndarray | None,
                             pos_rand_xyz: tuple[IGeneratorRandom | None,
                                                 IGeneratorRandom | None,
                                                 IGeneratorRandom | None] | None,

                            ) -> np.ndarray:
    sens_pos_perturbed = np.copy(sens_pos_nominal)

    if pos_offset_xyz is not None:
        sens_pos_perturbed = sens_pos_perturbed + pos_offset_xyz

    if pos_rand_xyz is not None:
        for ii,rng in enumerate(pos_rand_xyz):
            if rng is not None:
                sens_pos_perturbed[:,ii] = sens_pos_perturbed[:,ii] + \
                    rng.generate(size=sens_pos_perturbed.shape[0])

    return sens_pos_perturbed


def perturb_sample_times(sim_time: np.ndarray,
                         time_nominal: np.ndarray | None,
                         time_offset: np.ndarray | None,
                         time_rand: IGeneratorRandom | None,
                         time_drift: IDriftCalculator | None
                         ) -> np.ndarray | None:

    if time_nominal is None:
        if (time_offset is not None
            or time_rand is not None
            or time_drift is not None):
            time_nominal = sim_time
        else:
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


def perturb_sensor_angles(n_sensors: int,
                          angles_nominal: tuple[Rotation,...] | None,
                          angle_offsets_zyx: np.ndarray | None,
                          rand_ang_zyx: tuple[IGeneratorRandom | None,
                                              IGeneratorRandom | None,
                                              IGeneratorRandom | None] | None,
                          ) -> tuple[Rotation,...] | None:

    if angles_nominal is None:
        if angle_offsets_zyx is not None or rand_ang_zyx is not None:
            angles_nominal = n_sensors * \
                (Rotation.from_euler("zyx",[0,0,0], degrees=True),)
        else:
            return None

    angles_perturbed = [Rotation.from_euler("zyx",[0,0,0], degrees=True)] * \
        len(angles_nominal)
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

#-------------------------------------------------------------------------------
