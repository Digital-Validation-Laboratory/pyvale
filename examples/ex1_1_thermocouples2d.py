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
    """pyvale example: quick start example using the factory to generate a basic
    default thermocouple array and apply it to the simulation of a temperature
    field on a 2D plate.

    Shows how to use visualisation tools to view sensor locations on the
    simulation mesh using pyvista. Also shows how to plot point sensor traces
    using matplotlib.
    """

    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    field_name = list(sim_data.node_vars.keys())[0] # type: ignore

    # This creates a grid of 3x2 sensors in the xy plane
    n_sens = (3,2,1)    # Number of sensor (x,y,z)
    x_lims = (0.0,2.0)  # Limits for each coord in sim length units
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    # Gives a n_sensx3 array of sensor positions where each row is a sensor with
    # coords (x,y,z) - can also just manually create this array
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    # Now we create a thermocouple array with with the sensor positions and the
    # temperature field from the simulation
    tc_array = pyvale.SensorArrayFactory() \
        .basic_thermocouple_array(sim_data,
                                  sens_pos,
                                  field_name,
                                  spat_dims=2)

    # We can get an array of measurements from the sensor array for all time
    # steps in the simulation. A measurement is calculated as follows:
    # measurement = truth + systematic_error + random_error
    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor 0:\n{measurements[0,0,:]}\n')

    # We can also get the truth values, systematic and random errors as numpy
    # arrays
    truth_values = tc_array.get_truth_values()
    systematic_errs = tc_array.get_systematic_errs()
    random_errs = tc_array.get_random_errs()

    # Now we use pyvista to get a 3D interactive labelled plot of the sensor
    # locations on our simulation geometry.
    pv_plot = pyvale.plot_sensors(tc_array,field_name)
    # We label the temperature scale bar ourselves
    pv_plot.add_scalar_bar('Temperature, T [degC]')

    # Set this to 'interactive' to get an interactive 3D plot of the simulation
    # and labelled sensor locations, set to 'save_fig' to create a vector
    # graphic using a specified camera position.
    pv_plot_mode = 'interactive'

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
    (fig,_) = pyvale.plot_time_traces(tc_array,field_name)
    
    if trace_plot_mode == 'interactive':
        plt.show()
    if trace_plot_mode == 'save_fig':
        save_traces = Path('examples/images/plate_thermal_2d_traces.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
