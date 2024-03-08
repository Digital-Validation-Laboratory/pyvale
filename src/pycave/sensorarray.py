from pprint import pprint
from abc import ABC, abstractmethod
from typing import Callable, Any
from functools import partial
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import CellType

import mooseherder as mh

from pycave.plotprops import PlotProps


@dataclass
class MeasurementData():
    measurements: np.ndarray | None =  None
    random_errs: np.ndarray | None  = None
    systematic_errs: np.ndarray | None = None
    truth_values: np.ndarray | None = None


class SensorArray(ABC):
    @abstractmethod
    def get_positions(self) -> np.ndarray:
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
    def get_measurements(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_measurement_data(self) -> MeasurementData:
        pass


def create_sensor_pos_array(n_sens: tuple[int,int,int],
                           x_lims: tuple[float, float],
                           y_lims: tuple[float, float],
                           z_lims: tuple[float, float]) -> np.ndarray:

    sens_pos_x = np.linspace(x_lims[0],x_lims[1],n_sens[0]+2)[1:-1]
    sens_pos_y = np.linspace(y_lims[0],y_lims[1],n_sens[1]+2)[1:-1]
    sens_pos_z = np.linspace(z_lims[0],z_lims[1],n_sens[2]+2)[1:-1]

    (sens_grid_x,sens_grid_y,sens_grid_z) = np.meshgrid(
        sens_pos_x,sens_pos_y,sens_pos_z)

    sens_pos_x = sens_grid_x.flatten()
    sens_pos_y = sens_grid_y.flatten()
    sens_pos_z = sens_grid_z.flatten()

    sens_pos = np.vstack((sens_pos_x,sens_pos_y,sens_pos_z)).T
    return sens_pos


def plot_sensors(pv_simdata: pv.UnstructuredGrid,
                 pv_sensdata: pv.PolyData,
                 field_name: str,
                 time_step: int = -1) -> Any: # Stupid plotter doesn't allow type hinting!

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
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    pv_plot.add_axes_at_origin(labels_off=True)

    return pv_plot

