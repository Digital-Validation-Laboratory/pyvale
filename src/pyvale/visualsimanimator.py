'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import time
import copy
from pathlib import Path
import numpy as np
#import vtk #NOTE: has to be here to fix latex bug in pyvista/vtk
# See: https://github.com/pyvista/pyvista/discussions/2928
#NOTE: causes output to console to be suppressed unfortunately
import pyvista as pv

from pyvale.sensorarraypoint import SensorArrayPoint
from pyvale.visualplotopts import VisOptsSensorOnSim
from pyvale.visualsimplotter import plot_point_sensors_on_sim

def animate_sim_with_sensors(sensor_array: SensorArrayPoint,
                            component: str,
                            time_steps: np.ndarray | None = None,
                            vis_opts: VisOptsSensorOnSim | None = None) -> None:

    if vis_opts is None:
        vis_opts = VisOptsSensorOnSim()

    pv_plot = pv.Plotter(window_size=[1280, 800]) # type: ignore

    vis_sim = sensor_array.field.get_visualiser()
    descriptor = sensor_array.descriptor


    comp_ind = sensor_array.field.get_component_index(component)

    # Plot the simulation mesh
    pv_plot.add_mesh(vis_sim,
                     scalars="temperature",
                     label="sim-data",
                     show_edges=True,
                     show_scalar_bar=True,
                     scalar_bar_args={"title":descriptor.create_label(comp_ind),
                                      "vertical":True},
                     lighting=False)

    pv_plot.add_axes_at_origin(labels_off=True)

    #pv_plot.open_movie(Path().cwd() / 'dev' / 'test_output' / 'test.mp4')


    sim_scalars = np.copy(vis_sim[component])

    if time_steps is None:
        time_steps = sensor_array.get_sample_times()

    pv_plot.show(auto_close=False,interactive=False)
    pv_plot.render()

    for tt in range(time_steps.shape[0]):
        vis_sim[component] = sim_scalars[:,tt]

        pv_plot.add_text(f"Iteration: {tt}", name='time-label')
        pv_plot.render()
        time.sleep(0.1)

        print(80*"=")
        print(sim_scalars[:,tt])
        print(sim_scalars[:,tt].shape)
        print(80*"=")





