'''
================================================================================
pycave: mono repo

authors: thescepticalrabbit
================================================================================
'''
from abc import ABC, abstractmethod
from typing import Callable, Any
from functools import partial
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import CellType

import mooseherder as mh

from pycave.plotprops import PlotProps


#===============================================================================
def convert_simdata_to_pyvista(sim_data: mh.SimData, dim: int = 3
                               ) -> pv.UnstructuredGrid:

    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

    if sim_data.connect is None:
        raise RuntimeError("SimData does not have a connectivity table, unable to convert to pyvista")

    for cc in sim_data.connect:
        # NOTE: need the -1 here to make element numbers 0 indexed!
        temp_connect = sim_data.connect[cc]-1
        (nodes_per_elem,n_elems) = temp_connect.shape

        temp_connect = temp_connect.T.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)
        temp_connect = np.insert(temp_connect,idxs,nodes_per_elem)

        this_cell_type = get_cell_type(nodes_per_elem,dim=dim)
        cell_types = np.hstack((cell_types,np.full(n_elems,this_cell_type)))
        flat_connect = np.hstack((flat_connect,temp_connect),dtype=np.int64)


    cells = flat_connect
    points = sim_data.coords
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)

    return pv_grid

#===============================================================================
def attach_field_to_pyvista(pv_grid: pv.UnstructuredGrid,
                                 node_field: np.ndarray,
                                 name: str) -> pv.UnstructuredGrid:
    pv_grid[name] = node_field
    return pv_grid


#===============================================================================
def get_cell_type(nodes_per_elem: int, dim: int = 3) -> int:
    cell_type = 0

    if dim == 2:
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


#===============================================================================
class Field:
    def __init__(self, sim_data: mh.SimData, name: str, dim: int = 3) -> None:
        self._name = name
        self._data_grid = convert_simdata_to_pyvista(sim_data,dim)
        self._data_grid = attach_field_to_pyvista(self._data_grid,
                                                  sim_data.node_vars[name], # type: ignore
                                                  name)
        self._time_steps = sim_data.time

    def get_time_steps(self) -> np.ndarray:
        return self._time_steps # type: ignore

    def sample(self, sample_points: np.ndarray) -> np.ndarray:
        pv_points = pv.PolyData(sample_points)
        sample_data = pv_points.sample(self._data_grid)
        return np.array(sample_data[self._name]) # type: ignore

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._data_grid


#===============================================================================
@dataclass
class MeasurementData():
    measurements: np.ndarray | None =  None
    random_errs: np.ndarray | None  = None
    systematic_errs: np.ndarray | None = None
    truth_values: np.ndarray | None = None


#===============================================================================
class SensorArray(ABC):
    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_measurements(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_truth_values(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_systematic_errs(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_random_errs(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_measurement_data(self) -> MeasurementData:
        pass


#===============================================================================
class ThermocoupleArray(SensorArray):
    def __init__(self,
                 positions: np.ndarray,
                 field: Field) -> None:

        self._positions = positions
        self._field = field

        self._rand_err_func = None
        self._sys_err_func = None

        self._sensor_names = list([])
        for ss in range(self.get_num_sensors()):
            num_str = f'{ss}'.zfill(2)
            self._sensor_names.append(f'TC{num_str}')


    def get_positions(self) -> np.ndarray:
        return self._positions


    def get_num_sensors(self) -> int:
        return self._positions.shape[0]


    def get_measurement_shape(self) -> tuple[int,int]:
        return (self.get_num_sensors(),
                self._field.get_time_steps().shape[0])

    def get_sensor_names(self) -> list[str]:
        return self._sensor_names


    def get_measurements(self) -> np.ndarray:

        measurements = self.get_truth_values()
        sys_errs = self.get_systematic_errs()
        rand_errs = self.get_random_errs()

        if sys_errs is not None:
            measurements = measurements + sys_errs

        if rand_errs is not None:
            measurements = measurements + rand_errs

        return measurements


    def get_truth_values(self) -> np.ndarray:
        return self._field.sample(self._positions)


    def set_systematic_err_func(self, sys_fun: Callable | None = None) -> None:
        self._sys_err_func = sys_fun


    def get_systematic_errs(self) -> np.ndarray | None:
        if self._sys_err_func is None:
            return None

        return self._sys_err_func(size=self.get_measurement_shape())


    def set_random_err_func(self, rand_fun: Callable | None = None) -> None:
        self._rand_err_func = rand_fun


    def get_random_errs(self) -> np.ndarray | None:
        if self._rand_err_func is None:
            return None

        return self._rand_err_func(size=self.get_measurement_shape())


    def get_visualiser(self) -> pv.PolyData:
        pv_data = pv.PolyData(self._positions)
        pv_data['labels'] = self._sensor_names
        return pv_data


    def get_measurement_data(self) -> MeasurementData:
        measurement_data = MeasurementData()
        measurement_data.measurements = self.get_measurements()
        measurement_data.systematic_errs = self.get_systematic_errs()
        measurement_data.random_errs = self.get_random_errs()
        measurement_data.truth_values = self.get_truth_values()
        return measurement_data


    def plot_time_traces(self) -> tuple[Any,Any]:
        pp = PlotProps()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        fig, ax = plt.subplots(figsize=pp.single_fig_size,layout='constrained')
        fig.set_dpi(pp.resolution)

        p_time = self._field.get_time_steps()
        for ii in range(self.get_num_sensors()):
            measurements = self.get_measurements()
            ax.plot(p_time,measurements[ii,:],
                '-',label=self._sensor_names[ii],
                lw=pp.lw,ms=pp.ms,color=colors[ii])

        ax.set_xlabel('Time, $t$ [s]',
                    fontsize=pp.font_ax_size, fontname=pp.font_name)
        ax.set_ylabel('Temperature, $T$ [$\circ C$]',
                    fontsize=pp.font_ax_size, fontname=pp.font_name)

        plt.grid(True)
        ax.legend()
        ax.legend(prop={"size":pp.font_leg_size},loc='upper left')
        plt.draw()
        #plt.show()

        return (fig,ax)



#===============================================================================
def plot_sensors(pv_simdata: pv.UnstructuredGrid,
                 pv_sensdata: pv.PolyData,
                 field_name: str,
                 time_step: int = -1) -> Any: # Stupid plotter doesn't allow type hinting!
    #pv.set_plot_theme('dark') # type: ignore

    pv_plot = pv.Plotter(window_size=[1280, 800]) # type: ignore


    pv_plot.add_point_labels(pv_sensdata, "labels",
                            font_size=40,
                            shape_color='grey',
                            point_color='red',
                            render_points_as_spheres=True,
                            point_size=20,
                            always_visible=True
                            )

    pv_plot.add_mesh(pv_simdata,
                     scalars=pv_simdata[field_name][:,time_step],
                     label='sim data',
                     show_edges=True)

    pv_plot.add_axes_at_origin(labels_off=False)
    #pv_plot.set_scale(xscale = 100, yscale = 100, zscale = 100)
    #pv_plot.camera_position = 'xy'
    #pv_plot.camera.zoom(5)

    #pv_plot.show()

    return pv_plot
