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

from pyvale.field import (IField,
                                  conv_simdata_to_pyvista,
                                  sample_pyvista)

class FieldScalar(IField):
    __slots__ = ("_field_key","_spat_dims","_time_steps","_pyvista_gris")

    def __init__(self,
                 sim_data: mh.SimData,
                 field_key: str,
                 spat_dims: int) -> None:

        self._field_key = field_key
        self._spat_dims = spat_dims

        self._time_steps = sim_data.time
        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                                    (field_key,),
                                                    spat_dims)

    def set_sim_data(self, sim_data: mh.SimData) -> None:
        self._time_steps = sim_data.time
        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                            (self._field_key,),
                                            self._spat_dims)

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._pyvista_grid

    def get_all_components(self) -> tuple[str, ...]:
        return (self._field_key,)

    def get_component_index(self, comp: str) -> int:
        return 0 # scalar fields only have one component!

    def sample_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    angles: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:

        return sample_pyvista((self._field_key,),
                                self._pyvista_grid,
                                self._time_steps,
                                points,
                                times)

