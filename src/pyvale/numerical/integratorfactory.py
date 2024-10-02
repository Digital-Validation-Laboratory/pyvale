'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np

from pyvale.physics.field import IField
from pyvale.numerical.spatialintegrator import (ISpatialIntegrator,
                                                SpatialIntType)
from pyvale.numerical.rectangleintegrator import Rectangle2D
from pyvale.numerical.quadratureintegrator import (Quadrature2D,
                                                   create_gauss_weights_2d_4pts,
                                                   create_gauss_weights_2d_9pts)

class SpatialIntegratorFactory:
    @staticmethod
    def rect_2d_1pt(field: IField,
                    cent_pos: np.ndarray,
                    area_dims: np.ndarray,
                    sample_times: np.ndarray | None = None
                    )-> ISpatialIntegrator:

        int_pt_offsets = np.array([[0,0,0],])

        rect_int = Rectangle2D(int_pt_offsets,
                               field,
                               cent_pos,
                               area_dims,
                               sample_times)

        return rect_int


    @staticmethod
    def rect_2d_4pt(field: IField,
                    cent_pos: np.ndarray,
                    area_dims: np.ndarray,
                    sample_times: np.ndarray | None = None
                    )-> ISpatialIntegrator:

        int_pt_offsets = area_dims * np.array([[-0.5,-0.5,0],
                                          [-0.5,0.5,0],
                                          [0.5,-0.5,0],
                                          [0.5,0.5,0],])

        rect_int = Rectangle2D(int_pt_offsets,
                               field,
                               cent_pos,
                               area_dims,
                               sample_times)

        return rect_int


    @staticmethod
    def rect_2d_9pt(field: IField,
                    cent_pos: np.ndarray,
                    area_dims: np.ndarray,
                    sample_times: np.ndarray | None = None
                    )-> ISpatialIntegrator:

        int_pt_offsets = area_dims * np.array([[-1/3,-1/3,0],
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
                               cent_pos,
                               area_dims,
                               sample_times)

        return rect_int


    @staticmethod
    def quad_2d_4pt(field: IField,
                    cent_pos: np.ndarray,
                    area_dims: np.ndarray,
                    sample_times: np.ndarray | None = None
                    )-> ISpatialIntegrator:

        gauss_pt_offsets = area_dims * 1/np.sqrt(3)* np.array([[-1,-1,0],
                                                        [-1,1,0],
                                                        [1,-1,0],
                                                        [1,1,0]])

        gauss_weight_func = create_gauss_weights_2d_4pts

        quadrature = Quadrature2D(gauss_pt_offsets,
                            gauss_weight_func,
                            field,
                            cent_pos,
                            area_dims,
                            sample_times)
        return quadrature


    @staticmethod
    def quad_2d_9pt(field: IField,
                    cent_pos: np.ndarray,
                    area_dims: np.ndarray,
                    sample_times: np.ndarray | None = None
                    )-> ISpatialIntegrator:

        gauss_pt_offsets = area_dims * np.array([[-np.sqrt(0.6),-np.sqrt(0.6),0],
                                            [-np.sqrt(0.6),np.sqrt(0.6),0],
                                            [np.sqrt(0.6),-np.sqrt(0.6),0],
                                            [np.sqrt(0.6),np.sqrt(0.6),0],
                                            [-np.sqrt(0.6),0,0],
                                            [0,-np.sqrt(0.6),0],
                                            [0,np.sqrt(0.6),0],
                                            [np.sqrt(0.6),0,0],
                                            [0,0,0]])

        gauss_weight_func = create_gauss_weights_2d_9pts

        quadrature = Quadrature2D(gauss_pt_offsets,
                            gauss_weight_func,
                            field,
                            cent_pos,
                            area_dims,
                            sample_times)
        return quadrature


def build_spatial_integrator(integrator_type: SpatialIntType,
                            field: IField,
                            cent_pos: np.ndarray,
                            area_dims: np.ndarray,
                            sample_times: np.ndarray | None = None
                            ) -> ISpatialIntegrator:

    if integrator_type == SpatialIntType.RECT1PT:
        return SpatialIntegratorFactory.rect_2d_1pt(field,
                                                      cent_pos,
                                                      area_dims,
                                                      sample_times)
    elif integrator_type == SpatialIntType.RECT4PT:
        return SpatialIntegratorFactory.rect_2d_4pt(field,
                                                      cent_pos,
                                                      area_dims,
                                                      sample_times)
    elif integrator_type == SpatialIntType.RECT9PT:
        return SpatialIntegratorFactory.rect_2d_9pt(field,
                                                      cent_pos,
                                                      area_dims,
                                                      sample_times)
    elif integrator_type == SpatialIntType.QUAD4PT:
        return SpatialIntegratorFactory.quad_2d_4pt(field,
                                            cent_pos,
                                            area_dims,
                                            sample_times)
    elif integrator_type == SpatialIntType.QUAD9PT:
        return SpatialIntegratorFactory.quad_2d_9pt(field,
                                            cent_pos,
                                            area_dims,
                                            sample_times)