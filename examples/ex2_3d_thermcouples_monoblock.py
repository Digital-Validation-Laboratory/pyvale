'''
================================================================================
Example: 3d thermocouples on a monoblock

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pprint import pprint
from pathlib import Path

import matplotlib.pyplot as plt

import mooseherder as mh
import pyvale


def main() -> None:
    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('data/examplesims/monoblock_3d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    # Create a Field object that will allow the sensors to interpolate the sim
    # data field of interest quickly by using the mesh and shape functions
    spat_dims = 3       # Specify that we only have 2 spatial dimensions
    field_name = 'temperature'    # Same as in the moose input and SimData node_var key
    t_field = pyvale.ScalarField(sim_data,field_name,spat_dims)

    # This creates a grid of 3x2 sensors in the xy plane
    n_sens = (1,4,1)    # Number of sensor (x,y,z)
    x_lims = (11.5,11.5)  # Limits for each coord in sim length units
    y_lims = (-11.5,19.5)
    z_lims = (0.0,12.0)
    # Gives a n_sensx3 array of sensor positions where each row is a sensor with
    # coords (x,y,z) - can also just manually create this array
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    # Now we create a thermocouple array with with the sensor positions and the
    # temperature field from the simulation
    tc_array = pyvale.ThermocoupleArray(sens_pos,t_field)

    # Setup the UQ functions for the sensors. Here we use the basic defaults
    # which is a uniform distribution for the systematic error which is sampled
    # once and remains constant throughout the simulation time creating an
    # offset. The max temp in the simulation is ~800degC so this range [lo,hi]
    # should be visible on the time traces.
    err_sys1 = pyvale.SysErrUniform(low=-25.0,high=25.0)
    sys_err_int = pyvale.SysErrIntegrator([err_sys1],
                                          tc_array.get_measurement_shape())
    tc_array.set_sys_err_integrator(sys_err_int)

    # The default for the random error is a normal distribution here we specify
    # a standard deviation which should be visible on the time traces. Note that
    # the random error is sampled repeatedly for each time step.
    err_rand1 = pyvale.RandErrNormal(std=25.0)
    rand_err_int = pyvale.RandErrIntegrator([err_rand1],
                                            tc_array.get_measurement_shape())
    tc_array.set_rand_err_integrator(rand_err_int)

    # We can get an array of measurements as follows:
    measurements = tc_array.get_measurements()
    print(f'\nMeasurements:\n{measurements}\n')
    

    # Now we use pyvista to get a 3D interactive labelled plot of the sensor
    # locations on our simulation geometry.
    pv_sens = tc_array.get_visualiser()
    pv_sim = t_field.get_visualiser()
    pprint(pv_sim)

    pv_plot = pyvale.plot_sensors(pv_sim,pv_sens,field_name)
    # We label the temperature scale bar ourselves for clarity
    pv_plot.add_scalar_bar('Temp., T [degC]',vertical=True)

    # Set this to 'interactive' to get an interactive 3D plot of the simulation
    # and labelled sensor locations, set to 'save_fig' to create a vector
    # graphic using a specified camera position.
    pv_plot_mode = 'interactive'

    if pv_plot_mode == 'interactive':
        # Shows the pyvista interactive 3D plot
        pv_plot.camera_position = [(52.198, 26.042, 60.099),
                                    (0.0, 4.0, 5.5),
                                    (-0.190, 0.960, -0.206)]
        pv_plot.show()
        # Once the window is closed we plot the camera position to use later to
        # make a nice graphic for a paper/report
        pprint('Camera positions = ')
        pprint(pv_plot.camera_position)
    if pv_plot_mode == 'save_fig':
        # Determined manually by moving camera and then dumping camera position
        # to console after window close - see 'interactive above'
        pv_plot.camera_position = [(52.198, 26.042, 60.099),
                                    (0.0, 4.0, 5.5),
                                    (-0.190, 0.960, -0.206)]
        # Save a vector graphic to file for our selected camera view
        save_render = Path('examples/images/monoblock_thermal_sim_view.svg')
        pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
        pv_plot.screenshot(save_render.with_suffix('.png'))

    # Set this to 'interactive' to get a matplotlib.pyplot with the sensor
    # traces plotted over time. Set to 'save_fig' to save an image of the plot
    # to file.
    trace_plot_mode = 'interactive'

    # Plots the sensor time traces using matplotlib, thin solid lines are ground
    # truth from the simulation and dashed lines with '+' are simulated sensor
    # measurements using the specified UQ functions. The sensor traces should
    # have a uniform offset (systematic error) and noise (random error).
    (fig,_) = tc_array.plot_time_traces(plot_truth=True)
    if trace_plot_mode == 'interactive':
        plt.show()
    if trace_plot_mode == 'save_fig':
        save_traces = Path('examples/images/monoblock_thermal_traces.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
