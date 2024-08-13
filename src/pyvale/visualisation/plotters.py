'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
import pyvista as pv

import mooseherder as mh

from pyvale.physics.field import conv_simdata_to_pyvista
from pyvale.sensors.pointsensorarray import PointSensorArray
from pyvale.visualisation.plotopts import GeneralPlotOpts, SensorTraceOpts


def plot_sim_mesh(sim_data: mh.SimData) -> Any:
    pv_simdata = conv_simdata_to_pyvista(sim_data,
                                         None,
                                         sim_data.num_spat_dims)

    pv_plot = pv.Plotter(window_size=[1280, 800]) # type: ignore

    pv_plot.add_mesh(pv_simdata,
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    pv_plot.add_axes_at_origin(labels_off=True)
    return pv_plot

def plot_sim_data(sim_data: mh.SimData,
                  component: str,
                  time_step: int = -1) -> Any:
    pv_simdata = conv_simdata_to_pyvista(sim_data,
                                        (component,),
                                         sim_data.num_spat_dims)

    pv_plot = pv.Plotter(window_size=[1280, 800]) # type: ignore

    pv_plot.add_mesh(pv_simdata,
                     scalars=pv_simdata[component][:,time_step],
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    pv_plot.add_scalar_bar(component)
    pv_plot.add_axes_at_origin(labels_off=True)
    return pv_plot


def plot_sensors_on_sim(sensor_array: PointSensorArray,
                        component: str,
                        time_step: int = -1,
                        ) -> Any:

    pv_simdata = sensor_array.get_field().get_visualiser()
    pv_sensdata = sensor_array.get_visualiser()
    comp_ind = sensor_array.get_field().get_component_index(component)

    descriptor = sensor_array.get_descriptor()
    pv_sensdata['labels'] = descriptor.create_sensor_tags(
        sensor_array.get_measurement_shape()[0])

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
                     scalars=pv_simdata[component][:,time_step],
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    pv_plot.add_scalar_bar(descriptor.create_label(comp_ind),
                           vertical=True)
    pv_plot.add_axes_at_origin(labels_off=True)

    return pv_plot


def plot_time_traces(sensor_array: PointSensorArray,
                     component: str,
                     trace_opts: SensorTraceOpts | None = None,
                     plot_opts: GeneralPlotOpts | None = None
                     ) -> tuple[Any,Any]:

    field = sensor_array.get_field()
    comp_ind = sensor_array.get_field().get_component_index(component)
    samp_time = sensor_array.get_sample_times()
    measurements = sensor_array.get_measurements()
    n_sensors = sensor_array.get_positions().shape[0]
    descriptor = sensor_array.get_descriptor()

    if plot_opts is None:
        plot_opts = GeneralPlotOpts()

    if trace_opts is None:
        trace_opts = SensorTraceOpts()

    if trace_opts.sensors_to_plot is None:
        trace_opts.sensors_to_plot = np.arange(0,n_sensors)


    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    if trace_opts.sim_line is not None:
        sim_time = field.get_time_steps()
        sim_vals = field.sample_field(sensor_array.get_positions())


        for ss in range(n_sensors):
            if ss in trace_opts.sensors_to_plot:
                ax.plot(sim_time,
                        sim_vals[ss,comp_ind,:],
                        trace_opts.sim_line,
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ss % plot_opts.n_colors])

    if trace_opts.truth_line is not None:
        truth = sensor_array.get_truth_values()
        for ss in range(n_sensors):
            if ss in trace_opts.sensors_to_plot:
                ax.plot(samp_time,
                        truth[ss,comp_ind,:],
                        trace_opts.truth_line,
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ss % plot_opts.n_colors])

    sensor_tags = descriptor.create_sensor_tags(n_sensors)
    for ss in range(n_sensors):
        if ss in trace_opts.sensors_to_plot:
            ax.plot(samp_time,
                    measurements[ss,comp_ind,:],
                    trace_opts.meas_line,
                    label=sensor_tags[ss],
                    lw=plot_opts.lw,
                    ms=plot_opts.ms,
                    color=plot_opts.colors[ss % plot_opts.n_colors])

    ax.set_xlabel(trace_opts.time_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(descriptor.create_label(comp_ind),
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    if trace_opts.time_min_max is None:
        ax.set_xlim((np.min(samp_time),np.max(samp_time))) # type: ignore
    else:
        ax.set_xlim(trace_opts.time_min_max)

    if trace_opts.legend:
        ax.legend(prop={"size":plot_opts.font_leg_size},loc='best')

    plt.grid(True)
    plt.draw()

    return (fig,ax)
