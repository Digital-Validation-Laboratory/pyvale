'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
import pyvista as pv

import mooseherder as mh

from pyvale.physics.field import (IField,
                                  FieldError,
                                  conv_simdata_to_pyvista,
                                  sample_pyvista)

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

        #TODO: do some checking to make sure norm/dev components are consistent
        # based on the spatial dimensions

        if sim_data.time is None:
            raise(FieldError("SimData.time is None. SimData does not have time steps"))
        self._time_steps = sim_data.time

        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                            norm_components+dev_components,
                                            spat_dim)

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._pyvista_grid

    def get_all_components(self) -> tuple[str, ...]:
        return self._norm_components + self._dev_components

    def get_component_index(self, comp: str) -> int:
        return self.get_all_components().index(comp)

    def sample_field(self,
                sample_points: np.ndarray,
                sample_times: np.ndarray | None = None
                ) -> np.ndarray:

        return sample_pyvista(self._norm_components+self._dev_components,
                                self._pyvista_grid,
                                self._time_steps,
                                sample_points,
                                sample_times)

