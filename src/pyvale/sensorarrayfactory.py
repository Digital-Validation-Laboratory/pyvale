'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np

import mooseher

from pyvale.sensorarray import SensorArray
from pyvale.sensorlibrary.pointsensorarray import PointSensorArray
from pyvale.uncertainty.syserrintegrator import SysErrIntegrator
from pyvale.uncertainty.syserrcalculator import SysErrUniform
from pyvale.uncertainty.randerrintegrator import RandErrIntegrator
from pyvale.uncertainty.randerrcalculator import RandErrNormal

class SensorArrayFactory():
    def basic_thermocouple_array(sim_data: mh.SimData
                                 positions: np.ndarray,
                                 sample_times: np.ndarray | None = None
                                 ) -> SensorArray:


        sens_array = PointSensorArray(positions,,sample_times)

        err_sys1 = SysErrUniform(low=-20.0,high=20.0)
        sys_err_int = SysErrIntegrator([err_sys1],
                                        sens_array.get_measurement_shape())
        sens_array.set_sys_err_integrator(sys_err_int)

        err_rand1 = RandErrNormal(std=10.0)
        rand_err_int = RandErrIntegrator([err_rand1],
                                         sens_array.get_measurement_shape())
        sens_array.set_rand_err_integrator(rand_err_int)

        return sens_array
