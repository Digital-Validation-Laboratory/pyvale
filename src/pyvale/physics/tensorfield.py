'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
import mooseherder as mh

from pyvale.physics.field import (IField,
                                  conv_simdata_to_pyvista,
                                  sample_pyvista)
from pyvale.physics.coordtransform import (transform_tensor_2d,
                                           transform_tensor_3d)

class TensorField(IField):
    def __init__(self,
                 sim_data: mh.SimData,
                 field_key: str,
                 norm_components: tuple[str,...],
                 dev_components: tuple[str,...],
                 spat_dim: int) -> None:

        self._field_key = field_key
        self._norm_components = norm_components
        self._dev_components = dev_components
        self._spat_dim = spat_dim

        #TODO: do some checking to make sure norm/dev components are consistent
        # based on the spatial dimensions

        self._time_steps = sim_data.time
        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                            norm_components+dev_components,
                                            spat_dim)

    def set_sim_data(self, sim_data: mh.SimData) -> None:
        self._time_steps = sim_data.time
        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                            self._norm_components+
                                            self._dev_components,
                                            self._spat_dim)

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._pyvista_grid

    def get_all_components(self) -> tuple[str, ...]:
        return self._norm_components + self._dev_components

    def get_component_index(self, comp: str) -> int:
        return self.get_all_components().index(comp)

    def sample_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    orientations: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:

        field_data =  sample_pyvista(self._norm_components+self._dev_components,
                                    self._pyvista_grid,
                                    self._time_steps,
                                    points,
                                    times)

        if orientations is None:
            return field_data

        # NOTE:
        # ROTATION= object rotates with coords fixed
        # For Z rotation: sin negative in row 1.
        # TRANSFORMATION= coords rotate with object fixed
        # For Z transformation: sin negative in row 2, transpose scipy mat.

        #  Need to rotate each sensor using individual rotation = loop :(
        if self._spat_dim == 2:
            for ii,rr in enumerate(orientations):
                rmat = rr.as_matrix().T
                rmat = rmat[:2,:2]

                field_data[ii,:,:] = transform_tensor_2d(rmat,field_data[ii,:,:])

        else:
            for ii,rr in enumerate(orientations):
                rmat = rr.as_matrix().T

                field_data[ii,:,:] = transform_tensor_3d(rmat,field_data[ii,:,:])


        return field_data

