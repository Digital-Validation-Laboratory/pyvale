"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np

import mooseherder as mh

from pyvale.core.fieldscalar import FieldScalar
from pyvale.core.fieldvector import FieldVector
from pyvale.core.fieldtensor import FieldTensor
from pyvale.core.sensordescriptor import SensorDescriptorFactory
from pyvale.core.sensorarraypoint import SensorArrayPoint, SensorData
from pyvale.core.errorintegrator import ErrIntegrator
from pyvale.core.errorsysindep import ErrSysUniformPercent
from pyvale.core.errorrand import ErrRandNormPercent
from pyvale.core.errorsysdep import (ErrSysDigitisation,
                                     ErrSysSaturation)

#TODO: Docstrings

class SensorArrayFactory:
    """Namespace for static methods used to build common types of sensor arrays
    simplifying sensor array creation for users.
    """

    @staticmethod
    def thermocouples_no_errs(sim_data: mh.SimData,
                              sensor_data: SensorData,
                              field_name: str = "temperature",
                              spat_dims: int = 3,
                              ) -> SensorArrayPoint:
        """Builds and returns a point sensor array with common parameters used
        for thermocouples applied to a temperature field without any simulated
        measurement errors. Allows the user to build and attach their own error
        chain or use this for fast interpolation to sensor locations without
        errors.

        Parameters
        ----------
        sim_data : mh.SimData
            Simulation data containing a mesh and a temperature field for the
            thermocouple array to sample.
        sensor_data : SensorData
            _description_
        field_name : str, optional
            _description_, by default "temperature"
        spat_dims : int, optional
            , by default 3

        Returns
        -------
        SensorArrayPoint
            _description_
        """
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
                         sensor_data: SensorData,
                         sys_err_pc: float = 1.0,
                         rand_err_pc: float = 1.0) -> ErrIntegrator:
    """Builds a basic error integrator with uniform percentage systematic error
    calculator and a percentage normal random error calculator.

    Parameters
    ----------
    meas_shape : np.ndarray
        Shape of the measurement array which is (num_sensors,
        num_field_components,num_time_steps)
    sensor_data : SensorData
        Sensor array parameters for feeding through the error chain.
    sys_err_pc : float, optional
        Percentage systematic error, by default 1.0.
    rand_err_pc : float, optional
        Percentage random error, by default 1.0.

    Returns
    -------
    ErrIntegrator
        A basic error integrator with a uniform percentage systematic error and
        a normal percentage random error.
    """
    err_chain = []
    err_chain.append(ErrSysUniformPercent(-sys_err_pc,sys_err_pc))
    err_chain.append(ErrRandNormPercent(rand_err_pc))
    err_int = ErrIntegrator(err_chain,sensor_data,meas_shape)
    return err_int