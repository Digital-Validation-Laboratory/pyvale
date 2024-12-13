'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

import mooseherder as mh

from pyvale.core.fieldscalar import FieldScalar
from pyvale.core.fieldvector import FieldVector
from pyvale.core.fieldtensor import FieldTensor
from pyvale.core.sensordescriptor import SensorDescriptorFactory
from pyvale.core.sensorarraypoint import SensorArrayPoint, SensorData
from pyvale.core.errorintegrator import ErrIntegrator
from pyvale.errorsysindep import ErrSysUniformPercent
from pyvale.errorrand import ErrRandNormPercent
from pyvale.errorsysdep import (ErrSysDigitisation,
                                              ErrSysSaturation)


class SensorArrayFactory:
    @staticmethod
    def thermocouples_no_errs(sim_data: mh.SimData,
                              sensor_data: SensorData,
                              field_name: str = "temperature",
                              spat_dims: int = 3,
                              ) -> SensorArrayPoint:
        descriptor = SensorDescriptorFactory.temperature_descriptor()

        t_field = FieldScalar(sim_data,field_name,spat_dims)

        sens_array = SensorArrayPoint(sensor_data,
                                      t_field,
                                      descriptor)

        return sens_array

    @staticmethod
    def thermocouples_basic_errs(sim_data: mh.SimData,
                                 sensor_data: SensorData,
                                 field_name: str = "temperature",
                                 spat_dims: int = 3,
                                 errs_pc: float = 1.0
                                 ) -> SensorArrayPoint:

        sens_array = SensorArrayFactory.thermocouples_no_errs(sim_data,
                                                              sensor_data,
                                                              field_name,
                                                              spat_dims)

        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       sensor_data,
                                       errs_pc)

        # Normal thermcouple amp = 5mV / K
        err_int._err_chain.append(ErrSysDigitisation(bits_per_unit=2**16/1000))
        err_int._err_chain.append(ErrSysSaturation(meas_min=0.0,meas_max=1000.0))

        sens_array.set_error_integrator(err_int)
        return sens_array

    @staticmethod
    def disp_sensors_no_errs(sim_data: mh.SimData,
                            sensor_data: SensorData,
                            field_name: str = "displacement",
                            spat_dims: int = 3,
                            ) -> SensorArrayPoint:

        descriptor = SensorDescriptorFactory.displacement_descriptor()

        disp_field = FieldVector(sim_data,
                                 field_name,
                                 ('disp_x','disp_y'),
                                 spat_dims)

        sens_array = SensorArrayPoint(sensor_data,
                                      disp_field,
                                      descriptor)
        return sens_array


    @staticmethod
    def disp_sensors_basic_errs(sim_data: mh.SimData,
                                sensor_data: SensorData,
                                field_name: str = "displacement",
                                spat_dims: int = 3,
                                errs_pc: float = 1
                                ) -> SensorArrayPoint:

        sens_array = SensorArrayFactory.disp_sensors_no_errs(sim_data,
                                                            sensor_data,
                                                            field_name,
                                                            spat_dims)
        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       sensor_data,
                                       errs_pc)
        sens_array.set_error_integrator(err_int)

        return sens_array

    @staticmethod
    def strain_gauges_no_errs(sim_data: mh.SimData,
                              sensor_data: SensorData,
                              field_name: str = "strain",
                              spat_dims: int = 3
                              ) -> SensorArrayPoint:
        descriptor = SensorDescriptorFactory.strain_descriptor(spat_dims)

        if spat_dims == 2:
            norm_components = ('strain_xx','strain_yy')
            dev_components = ('strain_xy',)
        else:
            norm_components = ('strain_xx','strain_yy','strain_zz')
            dev_components = ('strain_xy','strain_yz','strain_xz')

        strain_field = FieldTensor(sim_data,
                                 field_name,
                                 norm_components,
                                 dev_components,
                                 spat_dims)

        sens_array = SensorArrayPoint(sensor_data,
                                      strain_field,
                                      descriptor)

        return sens_array


    @staticmethod
    def strain_gauges_basic_errs(sim_data: mh.SimData,
                                sensor_data: SensorData,
                                field_name: str = "strain",
                                spat_dims: int = 3,
                                errs_pc: float = 1.0
                                ) -> SensorArrayPoint:

        sens_array = SensorArrayFactory.strain_gauges_no_errs(sim_data,
                                                              sensor_data,
                                                              field_name,
                                                              spat_dims)

        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       sensor_data,
                                       errs_pc)
        sens_array.set_error_integrator(err_int)

        return sens_array


def basic_err_integrator(meas_shape: np.ndarray,
                         sensor_data,
                         errs_pc: float = 1.0) -> ErrIntegrator:
    err_chain = []
    err_chain.append(ErrSysUniformPercent(-errs_pc,errs_pc))
    err_chain.append(ErrRandNormPercent(errs_pc))
    err_int = ErrIntegrator(err_chain,sensor_data,meas_shape)
    return err_int