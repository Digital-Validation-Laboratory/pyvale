'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from abc import ABC, abstractmethod
import numpy as np
import pyvista as pv
from pyvista import CellType

import mooseherder as mh


class FieldError(Exception):
    pass


class Field(ABC):
    @abstractmethod
    def get_time_steps(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_visualiser(self) -> pv.UnstructuredGrid:
        pass

    @abstractmethod
    def get_all_components(self) -> tuple[str,...]:
        pass

    @abstractmethod
    def sample_field(self,
                    sample_points: np.ndarray,
                    sample_times: np.ndarray | None = None
                    ) -> np.ndarray:
        pass



# Needs to be able to return a scalar value at specified points
class ScalarField(Field):
    def __init__(self,
                 sim_data: mh.SimData,
                 field_name: str,
                 spat_dim: int) -> None:

        self._field_name = field_name

        if sim_data.time is None:
            raise(FieldError("SimData.time is None. SimData does not have time steps"))
        self._time_steps = sim_data.time

        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                                    (field_name,),
                                                    spat_dim)

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._pyvista_grid

    def get_all_components(self) -> tuple[str, ...]:
        return (self._field_name,)

    def sample_field(self,
                    sample_points: np.ndarray,
                    sample_times: np.ndarray | None = None
                    ) -> np.ndarray:

        return sample_pyvista((self._field_name,),
                                self._pyvista_grid,
                                self._time_steps,
                                sample_points,
                                sample_times)

# Need to be able to return vector values (with optional specified orientation)
# at specific points
# - Can assume given components are the normal ones but must be consistent with
#   the spatial dims
# AND
# - Need to deal with 2D and 3D spatial dims
class VectorField(Field):
    def __init__(self,
                 sim_data: mh.SimData,
                 field_name: str,
                 components: tuple[str,...],
                 spat_dim: int) -> None:

        self._field_name = field_name
        self._components = components

        self.all_components = self._components

        if sim_data.time is None:
            raise(FieldError("SimData.time is None. SimData does not have time steps"))
        self._time_steps = sim_data.time

        self._pyvista_grid = conv_simdata_to_pyvista(sim_data,
                                                    components,
                                                    spat_dim)

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._pyvista_grid

    def get_all_components(self) -> tuple[str, ...]:
        return self._components

    def sample_field(self,
                sample_points: np.ndarray,
                sample_times: np.ndarray | None = None
                ) -> np.ndarray:

        return sample_pyvista(self._components,
                                self._pyvista_grid,
                                self._time_steps,
                                sample_points,
                                sample_times)

# Need to be able to return tensor values (with optional specified orientation)
# at specific points
# - Need to know which are the normal components xx,yy,zz
# - Need to know which are the shear components xy,xz,yz
# AND
# - Need to deal with 2D and 3D spatial dims
class TensorField(Field):
    def __init__(self,
                 sim_data: mh.SimData,
                 field_name: str,
                 norm_components: tuple[str,...],
                 dev_components: tuple[str,...],
                 spat_dim: int) -> None:

        self._field_name = field_name
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

    def sample_field(self,
                sample_points: np.ndarray,
                sample_times: np.ndarray | None = None
                ) -> np.ndarray:

        return sample_pyvista(self._norm_components+self._dev_components,
                                self._pyvista_grid,
                                self._time_steps,
                                sample_points,
                                sample_times)
cell_type = CellType.QUAD
#-------------------------------------------------------------------------------
def conv_simdata_to_pyvista(sim_data: mh.SimData,
                            components: tuple[str,...],
                            spat_dim: int) -> pv.UnstructuredGrid:

    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

    if sim_data.connect is None:
        raise FieldError("SimData does not have a connectivity table, unable to convert to pyvista")
    if sim_data.node_vars is None:
        raise FieldError("SimData does not contain node_vars.")

    for cc in sim_data.connect:
        # NOTE: need the -1 here to make element numbers 0 indexed!
        temp_connect = sim_data.connect[cc]-1
        (nodes_per_elem,n_elems) = temp_connect.shape

        temp_connect = temp_connect.T.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)
        temp_connect = np.insert(temp_connect,idxs,nodes_per_elem)

        this_cell_type = get_cell_type(nodes_per_elem,spat_dim)
        cell_types = np.hstack((cell_types,np.full(n_elems,this_cell_type)))
        flat_connect = np.hstack((flat_connect,temp_connect),dtype=np.int64)

    cells = flat_connect
    points = sim_data.coords
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)

    for cc in components:
        pv_grid[cc] = sim_data.node_vars[cc]

    return pv_grid


def get_cell_type(nodes_per_elem: int, spat_dim: int) -> int:
    cell_type = 0

    if spat_dim == 2:
        if nodes_per_elem == 4:
            cell_type = CellType.QUAD
        elif nodes_per_elem == 3:
            cell_type = CellType.TRIANGLE
        else:
            cell_type = CellType.QUAD
    else:
        if nodes_per_elem == 8:
            cell_type =  CellType.HEXAHEDRON
        elif nodes_per_elem == 4:
            cell_type = CellType.TETRA
        else:
            cell_type = CellType.HEXAHEDRON

    return cell_type


def sample_pyvista(components: tuple,
                pyvista_grid: pv.UnstructuredGrid,
                time_steps: np.ndarray,
                sample_points: np.ndarray,
                sample_times: np.ndarray | None = None
                ) -> np.ndarray:

    pv_points = pv.PolyData(sample_points)
    sample_data = pv_points.sample(pyvista_grid)

    if sample_data is None:
        raise(FieldError("Sampling simulation data at sensors locations with pyvista failed."))

    n_comps = len(components)
    (n_sensors,n_time_steps) = np.array(sample_data[components[0]]).shape
    sample_at_sim_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_sim_time[:,ii,:] = np.array(sample_data[cc])

    if sample_times is None:
        return sample_at_sim_time

    sample_time_interp = lambda x: np.interp(sample_times,time_steps,x) # type: ignore

    n_time_steps = sample_times.shape[0]
    sample_at_spec_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_spec_time[:,ii,:] = np.apply_along_axis(sample_time_interp,1,
                                                        sample_at_sim_time[cc])

    return sample_at_spec_time
