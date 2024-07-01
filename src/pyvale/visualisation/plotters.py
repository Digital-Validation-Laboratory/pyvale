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
import pyvista as pv

from pyvale.sensors.pointsensorarray import PointSensorArray
from pyvale.visualisation.plotopts import GeneralPlotOpts, SensorPlotOpts
from pyvale.visualisation.plotopts import create_label_str, create_sensor_tags


def plot_sensors_on_sim(sensor_array: PointSensorArray,
                        field_to_plot: str,
                        time_step: int = -1,
                        plot_opts: SensorPlotOpts | None  = None
                        ) -> Any:

    pv_simdata = sensor_array.get_field().get_visualiser()
    pv_sensdata = sensor_array.get_visualiser()

    if plot_opts is None:
        plot_opts = SensorPlotOpts()

    descripts = sensor_array.get_descriptors()
    pv_sensdata['labels'] = create_sensor_tags(tag=descripts[3],
                                               n_sensors=sensor_array.get_measurement_shape()[0])

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
                     scalars=pv_simdata[field_to_plot][:,time_step],
                     label='sim-data',
                     show_edges=True,
                     show_scalar_bar=False)

    pv_plot.add_scalar_bar(create_label_str(descripts))
    pv_plot.add_axes_at_origin(labels_off=True)

    return pv_plot


def plot_time_traces(sensor_array: PointSensorArray,
                     component: str,
                     trace_props: SensorPlotOpts | None = None,
                     plot_opts: GeneralPlotOpts | None = None
                     ) -> tuple[Any,Any]:

    if plot_opts is None:
        plot_opts = GeneralPlotOpts()

    if trace_props is None:
        trace_props = SensorPlotOpts()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    field = sensor_array.get_field()
    comp_ind = sensor_array.get_field().get_component_index(component)

    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    if trace_props.sim_line is not None:
        sim_time = field.get_time_steps()
        sim_vals = field.sample_field(sensor_array.get_positions())

        for ii in range(sensor_array.get_positions().shape[0]):
            ax.plot(sim_time,sim_vals[ii,comp_ind,:],trace_props.sim_line,
                lw=plot_opts.lw/2,ms=plot_opts.ms/2,color=colors[ii])

    samp_time = sensor_array.get_sample_times()

    if trace_props.truth_line is not None:
        truth = sensor_array.get_truth_values()
        for ii in range(truth.shape[0]):
            ax.plot(samp_time,
                    truth[ii,comp_ind,:],
                    trace_props.truth_line,
                    lw=plot_opts.lw/2,
                    ms=plot_opts.ms/2,
                    color=colors[ii])

    measurements = sensor_array.get_measurements()
    for ii in range(measurements.shape[0]):
        ax.plot(samp_time,
                measurements[ii,comp_ind,:],
                trace_props.meas_line,
                label=trace_props.sensor_tag+str(ii),
                lw=plot_opts.lw/2,
                ms=plot_opts.ms/2,
                color=colors[ii])

    ax.set_xlabel(trace_props.time_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(trace_props.meas_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    ax.set_xlim([np.min(samp_time),np.max(samp_time)]) # type: ignore

    if trace_props.legend:
        ax.legend(prop={"size":plot_opts.font_leg_size},loc='best')

    plt.grid(True)
    plt.draw()

    return (fig,ax)