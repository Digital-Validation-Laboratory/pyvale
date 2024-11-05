'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np

from pyvale.physics.field import IField
from pyvale.sensors.sensordata import SensorData
from pyvale.numerical.spatialintegrator import ISpatialIntegrator
from pyvale.numerical.spatialinttype import ESpatialIntType
from pyvale.numerical.rectangleintegrator import Rectangle2D
from pyvale.numerical.quadratureintegrator import (Quadrature2D,
                                                   create_gauss_weights_2d_4pts,
                                                   create_gauss_weights_2d_9pts)

class SpatialIntegratorFactory:
    @staticmethod
    def rect_2d_1pt(field: IField,
                    sensor_data: SensorData,
                    )-> ISpatialIntegrator:

        int_pt_offsets = np.array([[0,0,0],])

        rect_int = Rectangle2D(int_pt_offsets,
                               field,
                               sensor_data.positions,
                               sensor_data.spatial_dims,
                               sensor_data.sample_times)

        return rect_int


    @staticmethod
    def rect_2d_4pt(field: IField,
                    sensor_data: SensorData,
                    )-> ISpatialIntegrator:

        int_pt_offsets = sensor_data.spatial_dims * np.array([[-0.5,-0.5,0],
                                                              [-0.5,0.5,0],
                                                              [0.5,-0.5,0],
                                                              [0.5,0.5,0],])

        rect_int = Rectangle2D(int_pt_offsets,
                               field,
                               sensor_data.positions,
                               sensor_data.spatial_dims,
                               sensor_data.sample_times)

        return rect_int


    @staticmethod
    def rect_2d_9pt(field: IField,
                    sensor_data: SensorData,
                    )-> ISpatialIntegrator:

        int_pt_offsets = sensor_data.spatial_dims * np.array([[-1/3,-1/3,0],
                                                              [-1/3,0,0],
                                                              [-1/3,1/3,0],
                                                              [0,-1/3,0],
                                                              [0,0,0],
                                                              [0,1/3,0],
                                                              [1/3,-1/3,0],
                                                              [1/3,0,0],
                                                              [1/3,1/3,0]])

        rect_int = Rectangle2D(int_pt_offsets,
                               field,
                               sensor_data.positions,
                               sensor_data.spatial_dims,
                               sensor_data.sample_times)

        return rect_int


    @staticmethod
    def quad_2d_4pt(field: IField,
                    sensor_data: SensorData,
                    )-> ISpatialIntegrator:

        gauss_pt_offsets = sensor_data.spatial_dims * 1/np.sqrt(3) * \
                                                np.array([[-1,-1,0],
                                                          [-1,1,0],
                                                          [1,-1,0],
                                                          [1,1,0]])

        gauss_weight_func = create_gauss_weights_2d_4pts

        return Quadrature2D(field,
                            sensor_data,
                            gauss_pt_offsets,
                            gauss_weight_func)


    @staticmethod
    def quad_2d_9pt(field: IField,
                    sensor_data: SensorData,
                    )-> ISpatialIntegrator:

        gauss_pt_offsets = sensor_data.spatial_dims * \
                                np.array([[-np.sqrt(0.6),-np.sqrt(0.6),0],
                                          [-np.sqrt(0.6),np.sqrt(0.6),0],
                                          [np.sqrt(0.6),-np.sqrt(0.6),0],
                                          [np.sqrt(0.6),np.sqrt(0.6),0],
                                          [-np.sqrt(0.6),0,0],
                                          [0,-np.sqrt(0.6),0],
                                          [0,np.sqrt(0.6),0],
                                          [np.sqrt(0.6),0,0],
                                          [0,0,0]])

        gauss_weight_func = create_gauss_weights_2d_9pts

        return Quadrature2D(field,
                            sensor_data,
                            gauss_pt_offsets,
                            gauss_weight_func)
    

def build_spatial_averager(field: IField, sensor_data: SensorData,
                          ) -> ISpatialIntegrator | None:
    if sensor_data.spatial_averager is None or sensor_data.spatial_dims is None:
        return None

    if sensor_data.spatial_averager == ESpatialIntType.RECT1PT:
        return SpatialIntegratorFactory.rect_2d_1pt(field,
                                                    sensor_data)
    elif sensor_data.spatial_averager == ESpatialIntType.RECT4PT:
        return SpatialIntegratorFactory.rect_2d_4pt(field,
                                                    sensor_data)
    elif sensor_data.spatial_averager == ESpatialIntType.RECT9PT:
        return SpatialIntegratorFactory.rect_2d_9pt(field,
                                                    sensor_data)
    elif sensor_data.spatial_averager == ESpatialIntType.QUAD4PT:
        return SpatialIntegratorFactory.quad_2d_4pt(field,
                                                    sensor_data)
    elif sensor_data.spatial_averager == ESpatialIntType.QUAD9PT:
        return SpatialIntegratorFactory.quad_2d_9pt(field,
                                                    sensor_data)