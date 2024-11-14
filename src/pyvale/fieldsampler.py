'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
from pyvale.field import IField
from pyvale.sensordata import SensorData
from pyvale.cameradata import CameraData
from pyvale.integratorfactory import build_spatial_averager

def sample_field_with_sensor_data(field: IField, sensor_data: SensorData
                                  ) -> np.ndarray:

    if sensor_data.spatial_averager is None:
        return field.sample_field(sensor_data.positions,
                                  sensor_data.sample_times,
                                  sensor_data.angles)

    spatial_integrator = build_spatial_averager(field,sensor_data)
    return spatial_integrator.calc_averages()

def sample_field_with_camera_data(field: IField, cam_data: CameraData
                                  ) -> np.ndarray:

    return field.sample_field()



