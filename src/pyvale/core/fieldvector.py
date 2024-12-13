'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
import mooseherder as mh

from pyvale.core.field import (IField,
                          conv_simdata_to_pyvista,
                          sample_pyvista)
from pyvale.core.fieldtransform import (transform_vector_2d,
                                   transform_vector_2d_batch,
                                   transform_vector_3d,
                                   transform_vector_3d_batch)

class FieldVector(IField):
    __slots__ = ("_field_key","_components","_spat_dims","_sim_data",
                 "_pyvista_grid","_pyvista_vis")

    def __init__(self,
                 sim_data: mh.SimData,
                 field_key: str,
                 components: tuple[str,...],
                 spat_dims: int) -> None:

        self._field_key = field_key
        self._components = components
        self._spat_dims = spat_dims

        self._sim_data = sim_data
        (self._pyvista_grid,self._pyvista_vis) = conv_simdata_to_pyvista(
            self._sim_data,
            self._components,
            self._spat_dims
        )

    def set_sim_data(self, sim_data: mh.SimData) -> None:
        self._sim_data = sim_data
        (self._pyvista_grid,self._pyvista_vis) = conv_simdata_to_pyvista(
            sim_data,
            self._components,
            self._spat_dims
        )

    def get_sim_data(self) -> mh.SimData:
        return self._sim_data

    def get_time_steps(self) -> np.ndarray:
        return self._sim_data.time

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._pyvista_vis

    def get_all_components(self) -> tuple[str, ...]:
        return self._components

    def get_component_index(self,comp: str) -> int:
        return self._components.index(comp)

    def sample_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    angles: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:

        field_data = sample_pyvista(self._components,
                                self._pyvista_grid,
                                self._sim_data.time,
                                points,
                                times)

        if angles is None:
            return field_data

        # NOTE:
        # ROTATION= object rotates with coords fixed
        # For Z rotation: sin negative in row 1.
        # TRANSFORMATION= coords rotate with object fixed
        # For Z transformation: sin negative in row 2, transpose scipy mat.

        # If we only have one angle we assume all sensors have the same angle
        # and we can batch process the rotations
        if len(angles) == 1:
            rmat = angles[0].as_matrix().T

            #TODO: assumes 2D in the x-y plane
            if self._spat_dims == 2:
                rmat = rmat[:2,:2]
                field_data = transform_vector_2d_batch(rmat,field_data)
            else:
                field_data = transform_vector_3d_batch(rmat,field_data)

        else: # Need to rotate each sensor using individual rotation = loop :(
            #TODO: assumes 2D in the x-y plane
            if self._spat_dims == 2:
                for ii,rr in enumerate(angles):
                    rmat = rr.as_matrix().T
                    rmat = rmat[:2,:2]
                    field_data[ii,:,:] = transform_vector_2d(rmat,field_data[ii,:,:])

            else:
                for ii,rr in enumerate(angles):
                    rmat = rr.as_matrix().T
                    field_data[ii,:,:] = transform_vector_3d(rmat,field_data[ii,:,:])

            #field_data[ii,:,:] = np.matmul(rmat,field_data[ii,:,:])
        return field_data

