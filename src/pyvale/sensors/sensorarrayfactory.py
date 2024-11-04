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
from pyvale.uncertainty.syserrorsIndep import SysErrUniformPercent
from pyvale.uncertainty.randerrors import RandErrNormPercent
from pyvale.uncertainty.syserrorsdep import (SysErrDigitisation,
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

        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       errs_pc)

        # Normal thermcouple amp = 5mV / K
        err_int._err_chain.append(SysErrDigitisation(bits_per_unit=2**16/1000))
        err_int._err_chain.append(SysErrSaturation(meas_min=0.0,meas_max=1000.0))

        sens_array.set_error_integrator(err_int)
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
        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       errs_pc)
        sens_array.set_error_integrator(err_int)

        return sens_array

    @staticmethod
    def strain_gauges_no_errs(sim_data: mh.SimData,
                              sensor_array_data: SensorData,
                              field_name: str = "strain",
                              spat_dims: int = 3
                              ) -> PointSensorArray:
        descriptor = SensorDescriptorFactory.strain_descriptor(spat_dims)

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

        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       errs_pc)
        sens_array.set_error_integrator(err_int)

        return sens_array


def basic_err_integrator(meas_shape: np.ndarray,
                        errs_pc: float = 1.0) -> ErrorIntegrator:
    err_chain = []
    err_chain.append(SysErrUniformPercent(-errs_pc,errs_pc))
    err_chain.append(RandErrNormPercent(errs_pc))
    err_int = ErrorIntegrator(err_chain,meas_shape)
    return err_int