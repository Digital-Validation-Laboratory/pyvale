'''
================================================================================
pycave: the python computer aided validation engine.
license: LGPL-2.1
Copyright (C) 2024 Lloyd Fletcher (scepticalrabbit)
================================================================================
'''
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from pycave.field import Field
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


@dataclass
class TraceProps:
    legend: bool = True
    y_label: str = r'Sensor Value, [unit]'
    x_label: str = r'Time, $t$ [s]'
    truth_line: str | None = '-'
    sim_line: str | None = '-'
    meas_line: str = '--+'
    sensors: np.ndarray | None = None
    time_inds: np.ndarray | None = None


def plot_time_traces(sensor_array: SensorArray,
                     field: Field | None = None,
                     trace_props: TraceProps | None  = None,
                     plot_props: PlotProps | None = None
                     ) -> tuple[Any,Any]:

        if plot_props is None:
            plot_props = PlotProps()

        if trace_props is None:
            trace_props = TraceProps()

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        fig, ax = plt.subplots(figsize=plot_props.single_fig_size,layout='constrained')
        fig.set_dpi(plot_props.resolution)


        if field is not None and trace_props.sim_line is not None:
            sim_time = field.get_time_steps()
            sim_vals = field.sample(sensor_array.get_positions())
            for ii in range(sensor_array.get_positions()[0]):
                ax.plot(sim_time,sim_vals[ii,:],'-o',
                    lw=plot_props.lw/2,ms=plot_props.ms/2,color=colors[ii])

        samp_time = sensor_array.get_sample_times()

        if trace_props.truth_line is not None:
            truth = sensor_array.get_truth_values()
            for ii in range(truth.shape[0]):
                ax.plot(samp_time,
                        truth[ii,:],
                        trace_props.truth_line,
                        lw=plot_props.lw/2,
                        ms=plot_props.ms/2,
                        color=colors[ii])

        measurements = sensor_array.get_measurements()
        for ii in range(measurements.shape[0]):
            ax.plot(samp_time,
                    measurements[ii,:],
                    trace_props.meas_line,
                    #label=self._sensor_names[ii],
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


