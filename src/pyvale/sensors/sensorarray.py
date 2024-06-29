'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from pyvale.field import Field
from pyvale.plotprops import PlotProps
from pyvale.plotprops import SensorPlotProps

@dataclass
class MeasurementData():
    measurements: np.ndarray | None =  None
    random_errs: np.ndarray | None  = None
    systematic_errs: np.ndarray | None = None
    truth_values: np.ndarray | None = None


class SensorArray(ABC):
    @abstractmethod
    def get_field(self) -> Field:
        pass

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_sample_times(self) -> np.ndarray:
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

    @abstractmethod
    def get_visualiser(self) -> pv.PolyData:
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


def plot_sensors(sensor_array: SensorArray,
                 field_name: str,
                 time_step: int = -1,
                 plot_props: SensorPlotProps | None  = None) -> Any: # plotter doesn't allow type hinting!

    pv_simdata = sensor_array.get_field().get_visualiser()
    pv_sensdata = sensor_array.get_visualiser()

    if plot_props is None:
        plot_props = SensorPlotProps()

    sensor_names = list()
    for ss in range(pv_sensdata.n_points):
        num_str = f'{ss}'.zfill(2)
        sensor_names.append(f'{plot_props.sensor_tag}{num_str}')

    pv_sensdata['labels'] = sensor_names

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



def plot_time_traces(sensor_array: SensorArray,
                     component: str,
                     trace_props: SensorPlotProps | None  = None,
                     plot_props: PlotProps | None = None
                     ) -> tuple[Any,Any]:

    if plot_props is None:
        plot_props = PlotProps()

    if trace_props is None:
        trace_props = SensorPlotProps()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    field = sensor_array.get_field()
    comp_ind = sensor_array.get_field().get_component_index(component)

    fig, ax = plt.subplots(figsize=plot_props.single_fig_size,layout='constrained')
    fig.set_dpi(plot_props.resolution)

    if trace_props.sim_line is not None:
        sim_time = field.get_time_steps()
        sim_vals = field.sample_field(sensor_array.get_positions())

        for ii in range(sensor_array.get_positions().shape[0]):
            ax.plot(sim_time,sim_vals[ii,comp_ind,:],trace_props.sim_line,
                lw=plot_props.lw/2,ms=plot_props.ms/2,color=colors[ii])

    samp_time = sensor_array.get_sample_times()

    if trace_props.truth_line is not None:
        truth = sensor_array.get_truth_values()
        for ii in range(truth.shape[0]):
            ax.plot(samp_time,
                    truth[ii,comp_ind,:],
                    trace_props.truth_line,
                    lw=plot_props.lw/2,
                    ms=plot_props.ms/2,
                    color=colors[ii])

    measurements = sensor_array.get_measurements()
    for ii in range(measurements.shape[0]):
        ax.plot(samp_time,
                measurements[ii,comp_ind,:],
                trace_props.meas_line,
                label=trace_props.sensor_tag+str(ii),
                lw=plot_props.lw/2,
                ms=plot_props.ms/2,
                color=colors[ii])

    ax.set_xlabel(trace_props.x_label,
                fontsize=plot_props.font_ax_size, fontname=plot_props.font_name)
    ax.set_ylabel(trace_props.y_label,
                fontsize=plot_props.font_ax_size, fontname=plot_props.font_name)

    ax.set_xlim([np.min(samp_time),np.max(samp_time)]) # type: ignore

    if trace_props.legend:
        ax.legend(prop={"size":plot_props.font_leg_size},loc='best')

    plt.grid(True)
    plt.draw()

    return (fig,ax)


def print_measurements(sens_array: SensorArray,
                       sensors: tuple[int,int],
                       components: tuple[int,int],
                       time_steps: tuple[int,int])  -> None:

    measurement =  sens_array.get_measurements()
    truth = sens_array.get_truth_values()
    sys_errs = sens_array.get_systematic_errs()
    rand_errs = sens_array.get_random_errs()

    print(f"\nmeasurement.shape = \n    {measurement.shape}")
    print(f"measurement = \n    {measurement[sensors[0]:sensors[1],
                                             components[0]:components[1],
                                             time_steps[0]:time_steps[1]]}")
    print(f"truth = \n    {truth[sensors[0]:sensors[1],
                                components[0]:components[1],
                                time_steps[0]:time_steps[1]]}")
    if sys_errs is not None:
        print(f"sys_errs = \n    {sys_errs[sensors[0]:sensors[1],
                                            components[0]:components[1],
                                            time_steps[0]:time_steps[1]]}")
    if rand_errs is not None:
        print(f"rand_errs = \n    {rand_errs[sensors[0]:sensors[1],
                                            components[0]:components[1],
                                            time_steps[0]:time_steps[1]]}")
    print()

