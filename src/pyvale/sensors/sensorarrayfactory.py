'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np

import mooseherder as mh

from pyvale.physics.scalarfield import ScalarField
from pyvale.physics.vectorfield import VectorField
from pyvale.physics.tensorfield import TensorField
from pyvale.sensors.sensordescriptor import SensorDescriptorFactory
from pyvale.sensors.pointsensorarray import PointSensorArray, SensorData
from pyvale.uncertainty.errorintegrator import ErrorIntegrator
from pyvale.uncertainty.syserrors import SysErrUniformPercent
from pyvale.uncertainty.randerrors import RandErrNormPercent
from pyvale.uncertainty.depsyserrors import (SysErrDigitisation,
                                              SysErrSaturation)


class SensorArrayFactory:
    @staticmethod
    def thermocouples_no_errs(sim_data: mh.SimData,
                              sensor_array_data: SensorData,
                              field_name: str = "temperature",
                              spat_dims: int = 3,
                              ) -> PointSensorArray:
        descriptor = SensorDescriptorFactory.temperature_descriptor()

        t_field = ScalarField(sim_data,field_name,spat_dims)

        sens_array = PointSensorArray(sensor_array_data,
                                      t_field,
                                      descriptor)

        return sens_array

    @staticmethod
    def thermocouples_basic_errs(sim_data: mh.SimData,
                                 sensor_array_data: SensorData,
                                 field_name: str = "temperature",
                                 spat_dims: int = 3,
                                 errs_pc: float = 1.0
                                 ) -> PointSensorArray:

        sens_array = SensorArrayFactory.thermocouples_no_errs(sim_data,
                                                              sensor_array_data,
                                                              field_name,
                                                              spat_dims)

        sens_array = init_basic_errs(sens_array,errs_pc)

        # Normal thermcouple amp = 5mV / K
        dep_sys_err1 = SysErrDigitisation(bits_per_unit=2**16/1000)
        dep_sys_err2 = SysErrSaturation(meas_min=0.0,meas_max=1000.0)
        dep_sys_err_int = ErrorIntegrator([dep_sys_err1,dep_sys_err2],
                                            sens_array.get_measurement_shape())
        sens_array.set_systematic_err_integrator_dependent(dep_sys_err_int)

        return sens_array

    @staticmethod
    def disp_sensors_no_errs(sim_data: mh.SimData,
                            sensor_array_data: SensorData,
                            field_name: str = "displacement",
                            spat_dims: int = 3,
                            ) -> PointSensorArray:

        descriptor = SensorDescriptorFactory.displacement_descriptor()

        disp_field = VectorField(sim_data,
                                 field_name,
                                 ('disp_x','disp_y'),
                                 spat_dims)

        sens_array = PointSensorArray(sensor_array_data,
                                      disp_field,
                                      descriptor)
        return sens_array


    @staticmethod
    def disp_sensors_basic_errs(sim_data: mh.SimData,
                                sensor_array_data: SensorData,
                                field_name: str = "displacement",
                                spat_dims: int = 3,
                                errs_pc: float = 1
                                ) -> PointSensorArray:

        sens_array = SensorArrayFactory.disp_sensors_no_errs(sim_data,
                                                            sensor_array_data,
                                                            field_name,
                                                            spat_dims)
        sens_array = init_basic_errs(sens_array,errs_pc)

        return sens_array

    @staticmethod
    def strain_gauges_no_errs(sim_data: mh.SimData,
                              sensor_array_data: SensorData,
                              field_name: str = "strain",
                              spat_dims: int = 3
                              ) -> PointSensorArray:
        descriptor = SensorDescriptorFactory.strain_descriptor()

        if spat_dims == 2:
            norm_components = ('strain_xx','strain_yy')
            dev_components = ('strain_xy',)
        else:
            norm_components = ('strain_xx','strain_yy','strain_zz')
            dev_components = ('strain_xy','strain_yz','strain_xz')

        strain_field = TensorField(sim_data,
                                 field_name,
                                 norm_components,
                                 dev_components,
                                 spat_dims)

        sens_array = PointSensorArray(sensor_array_data,
                                      strain_field,
                                      descriptor)

        return sens_array


    @staticmethod
    def strain_gauges_basic_errs(sim_data: mh.SimData,
                                sensor_array_data: SensorData,
                                field_name: str = "strain",
                                spat_dims: int = 3,
                                errs_pc: float = 1.0
                                ) -> PointSensorArray:

        sens_array = SensorArrayFactory.strain_gauges_no_errs(sim_data,
                                                              sensor_array_data,
                                                              field_name,
                                                              spat_dims)
        sens_array = init_basic_errs(sens_array,errs_pc)

        return sens_array


def init_basic_errs(sens_array: PointSensorArray,
                    errs_pc: float = 1.0) -> PointSensorArray:

    indep_sys_err_int = ErrorIntegrator([SysErrUniformPercent(-errs_pc,errs_pc)],
                                    sens_array.get_measurement_shape())
    sens_array.set_systematic_err_integrator_independent(indep_sys_err_int)

    rand_err_int = ErrorIntegrator([RandErrNormPercent(errs_pc)],
                                        sens_array.get_measurement_shape())
    sens_array.set_random_err_integrator(rand_err_int)

    return sens_array