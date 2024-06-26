'''
================================================================================
example: thermocouples on a 2d plate

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
    """main
    """
    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    # Create a Field object that will allow the sensors to interpolate the sim
    # data field of interest quickly by using the mesh and shape functions
    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'temperature'    # Same as in the moose input and SimData node_var key
    t_field = pyvale.ScalarField(sim_data,field_name,spat_dims)

    # This creates a grid of 3x2 sensors in the xy plane
    n_sens = (3,2,1)    # Number of sensor (x,y,z)
    x_lims = (0.0,2.0)  # Limits for each coord in sim length units
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    # Gives a n_sensx3 array of sensor positions where each row is a sensor with
    # coords (x,y,z) - can also just manually create this array
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    # Now we create a point sensor array with with the sensor positions and the
    # temperature field from the simulation
    tc_array = pyvale.PointSensorArray(sens_pos,t_field)

    # Setup the UQ functions for the sensors. Here we can specify a list of
    # objects that will each calculate an error which will be added together
    # (integrated).
    err_sys1 = pyvale.SysErrUniform(low=-20.0,high=20.0)
    err_sys2 = pyvale.SysErrNormal(std=20.0)
    sys_err_int = pyvale.ErrorIntegrator([err_sys1,err_sys2],
                                          tc_array.get_measurement_shape())
    tc_array.set_sys_err_integrator(sys_err_int)

    # Random errors are also integrated and we can chain objects to calculate
    # multiple random error functions. The random error is sampled repeatedly
    # for each time step.
    err_rand1 = pyvale.RandErrNormal(std=10.0)
    err_rand2 = pyvale.RandErrUniform(low=-10.0,high=10.0)
    rand_err_int = pyvale.ErrorIntegrator([err_rand1,err_rand2],
                                            tc_array.get_measurement_shape())
    tc_array.set_rand_err_integrator(rand_err_int)


    # Now we use pyvista to get a 3D interactive labelled plot of the sensor
    # locations on our simulation geometry.
    pv_plot = pyvale.plot_sensors(tc_array,field_name)
    # We label the temperature scale bar ourselves
    pv_plot.add_scalar_bar('Temperature, T [degC]')

    # Set this to 'interactive' to get an interactive 3D plot of the simulation
    # and labelled sensor locations, set to 'save_fig' to create a vector
    # graphic using a specified camera position.
    pv_plot_mode = 'off'

    if pv_plot_mode == 'interactive':
        # Shows the pyvista interactive 3D plot
        pv_plot.show()
        # Once the window is closed we plot the camera position to use later to
        # make a nice graphic for a paper/report
        pprint('Camera positions = ')
        pprint(pv_plot.camera_position)
    if pv_plot_mode == 'save_fig':
        # Determined manually by moving camera and then dumping camera position
        # to console after window close - see 'interactive above'
        pv_plot.camera_position = [(-0.295, 1.235, 3.369),
                                (1.0274, 0.314, 0.0211),
                                (0.081, 0.969, -0.234)]
        # Save a vector graphic to file for our selected camera view
        save_render = Path('examples/images/plate_thermal_2d_sim_view.svg')
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
    (fig,_) = pyvale.plot_time_traces(tc_array,
                                      field_name)
    if trace_plot_mode == 'interactive':
        plt.show()
    if trace_plot_mode == 'save_fig':
        save_traces = Path('examples/images/plate_thermal_2d_traces.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
