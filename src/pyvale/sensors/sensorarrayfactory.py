'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

import mooseherder as mh

from pyvale.field import ScalarField
from pyvale.sensors.pointsensorarray import PointSensorArray
from pyvale.uncertainty.errorintegrator import ErrorIntegrator
from pyvale.uncertainty.presyserrors import SysErrUniform

from pyvale.uncertainty.randerrors import RandErrNormal

class SensorArrayFactory():
    def basic_thermocouple_array(self,
                                 sim_data: mh.SimData,
                                 positions: np.ndarray,
                                 field_name: str = "temperature",
                                 spat_dims: int = 3,
                                 sample_times: np.ndarray | None = None
                                 ) -> PointSensorArray:

        t_field = ScalarField(sim_data,field_name,spat_dims)

        sens_array = PointSensorArray(positions,t_field,sample_times)

        err_sys1 = SysErrUniform(low=-10.0,high=10.0)
        sys_err_int = ErrorIntegrator([err_sys1],
                                        sens_array.get_measurement_shape())
        sens_array.set_sys_err_integrator(sys_err_int)

        err_rand1 = RandErrNormal(std=10.0)
        rand_err_int = ErrorIntegrator([err_rand1],
                                         sens_array.get_measurement_shape())
        sens_array.set_rand_err_integrator(rand_err_int)

        return sens_array
