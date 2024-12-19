"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
from pyvale.core.field import IField
from pyvale.core.sensordata import SensorData
from pyvale.core.integratorfactory import build_spatial_averager

def sample_field_with_sensor_data(field: IField, sensor_data: SensorData
                                  ) -> np.ndarray:

    if sensor_data.spatial_averager is None:
        return field.sample_field(sensor_data.positions,
                                  sensor_data.sample_times,
                                  sensor_data.angles)

    spatial_integrator = build_spatial_averager(field,sensor_data)
    return spatial_integrator.calc_averages()





