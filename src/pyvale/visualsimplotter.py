'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from typing import Any
#import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
import pyvista as pv

import mooseherder as mh

from pyvale.field import conv_simdata_to_pyvista
from pyvale.sensorarraypoint import SensorArrayPoint
from pyvale.visualplotopts import VisOptsSensorOnSim


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
                     label="sim-data",
                     show_edges=True,
                     show_scalar_bar=True,
                     scalar_bar_args={"title":component},)

    pv_plot.add_axes_at_origin(labels_off=True)
    return pv_plot


def plot_point_sensors_on_sim(sensor_array: SensorArrayPoint,
                              component: str,
                              time_step: int = -1,
                              vis_opts: VisOptsSensorOnSim | None = None
                              ) -> pv.Plotter:

    if vis_opts is None:
        vis_opts = VisOptsSensorOnSim()

    pv_plot = pv.Plotter(window_size=[1280, 800]) # type: ignore

    vis_sim = sensor_array.field.get_visualiser()
    sim_data = sensor_array.field.get_sim_data()
    print(80*"=")
    print(sim_data.node_vars[component][:,time_step].shape)
    print(80*"=")
    vis_sim[component] = sim_data.node_vars[component][:,time_step]

    descriptor = sensor_array.descriptor

    vis_sens_nominal = pv.PolyData(sensor_array.sensor_data.positions)
    sens_data_perturbed = sensor_array.get_sensor_data_perturbed()

    if sens_data_perturbed is not None:
        vis_sens_perturbed = pv.PolyData(sens_data_perturbed.positions)
        vis_sens_perturbed["labels"] = ["",]*sensor_array.get_measurement_shape()[0]

    else:
        vis_sens_perturbed = None

    comp_ind = sensor_array.field.get_component_index(component)

    vis_sens_nominal["labels"] = descriptor.create_sensor_tags(
        sensor_array.get_measurement_shape()[0])

    # Add points to show sensor locations
    pv_plot.add_point_labels(vis_sens_nominal,"labels",
                            font_size=35,
                            shape_color="grey",
                            point_color="red",
                            render_points_as_spheres=True,
                            point_size=20,
                            always_visible=True)

    if vis_sens_perturbed is not None and vis_opts.show_perturbed_pos:
        pv_plot.add_point_labels(vis_sens_perturbed,"labels",
                                font_size=35,
                                shape_color="grey",
                                point_color="blue",
                                render_points_as_spheres=True,
                                point_size=20,
                                always_visible=True)

    # Plot the simulation mesh
    pv_plot.add_mesh(vis_sim,
                     scalars=component,
                     label="sim-data",
                     show_edges=True,
                     show_scalar_bar=True,
                     scalar_bar_args={"title":descriptor.create_label(comp_ind),
                                      "vertical":True},
                     lighting=False)

    pv_plot.add_axes_at_origin(labels_off=True)

    return pv_plot

