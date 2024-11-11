'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import time

from pathlib import Path
import numpy as np
#import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
import pyvista as pv

from pyvale.sensorarraypoint import SensorArrayPoint
from pyvale.visualplotopts import VisOptsSimAndSensors, VisOptsAnimation
from pyvale.visualsimplotter import plot_point_sensors_on_sim

def animate_sim_with_sensors(sensor_array: SensorArrayPoint,
                            component: str,
                            time_steps: np.ndarray | None = None,
                            vis_opts: VisOptsSimAndSensors | None = None,
                            anim_opts: VisOptsAnimation | None = None,
                            ) -> pv.Plotter:

    if vis_opts is None:
        vis_opts = VisOptsSimAndSensors()

    if anim_opts is None:
        anim_opts = VisOptsAnimation()

    if time_steps is None:
        time_steps = np.arange(0,sensor_array.get_sample_times().shape[0])

    sim_vis = sensor_array.field.get_visualiser()
    sim_data = sensor_array.field.get_sim_data()
    sim_vis[component] = sim_data.node_vars[component][:,0]

    if vis_opts.colour_bar_lims is None:
        min_comp = np.min(sim_data.node_vars[component][:,time_steps].flatten())
        max_comp = np.max(sim_data.node_vars[component][:,time_steps].flatten())
        vis_opts.colour_bar_lims = (min_comp,max_comp)

    descriptor = sensor_array.descriptor
    comp_ind = sensor_array.field.get_component_index(component)

    # Create the plotter
    pv_plot = pv.Plotter(window_size=vis_opts.window_size_px)

    # Set plotter options before adding sim
    pv_plot.set_background(vis_opts.background_colour)
    pv.global_theme.font.color = "white"
    pv_plot.add_axes_at_origin(labels_off=True)

    # Add the simulation data to the
    pv_plot.add_mesh(sim_vis,
                     scalars=component,
                     label="sim-data",
                     show_edges=vis_opts.show_edges,
                     show_scalar_bar=vis_opts.colour_bar_show,
                     scalar_bar_args={"title":descriptor.create_label(comp_ind),
                                      "vertical":True,
                                      "title_font_size":vis_opts.colour_bar_font_size,
                                      "label_font_size":vis_opts.colour_bar_font_size,
                                      },
                     lighting=False,
                     clim=vis_opts.colour_bar_lims)

    # Set plotter options post sim add (allows camera position to scale easily)
    pv_plot.camera_position = vis_opts.camera_position
    pv_plot.show(auto_close=False,interactive=False)

    for tt in time_steps:
        # Update the field plotted on the mesh
        sim_vis[component] = sim_data.node_vars[component][:,tt]

        if vis_opts.time_label_show:
            pv_plot.add_text(f"Time: {sim_data.time[tt]} {descriptor.time_units}",
                             position=vis_opts.time_label_position,
                             font_size=vis_opts.time_label_font_size,
                             name='time-label')

        pv_plot.render()
        time.sleep(1/anim_opts.frames_per_second)

    # Allow the user to interact with the plot after plotting the animation
    pv_plot.show(auto_close=False,interactive=True)

    return pv_plot






