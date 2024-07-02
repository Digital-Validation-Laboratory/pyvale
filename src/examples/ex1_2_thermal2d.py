'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    """pyvale example: building a point sensor array applied to a scalar field
    (temperature) from scratch. Also outlines the sensor conceptual model and
    how additional experiments can be generated.

    A sensor measurement is defined as:
    measurement = truth + systematic error + random error.

    Calling the 'get' methods of the sensor array will retrieve the results for
    the current experiment. Calling the 'calc' methods will generate a new
    experiment by sampling/calculating the systematic and random errors. The
    'calc' method must be called to initialise the errors.
    """
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    descriptor = pyvale.SensorDescriptor()
    descriptor.name = 'Temperature'
    descriptor.symbol = 'T'
    descriptor.units = r'^{\circ}C'
    descriptor.tag = 'TC'

    spat_dims = 2
    field_key = 'temperature'
    t_field = pyvale.ScalarField(sim_data,field_key,spat_dims)

    n_sens = (5,2,1)
    x_lims = (0.0,2.0)
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.PointSensorArray(sens_pos,t_field)

    err_sys1 = pyvale.SysErrUniform(low=-20.0,high=20.0)
    err_sys2 = pyvale.SysErrNormal(std=20.0)
    sys_err_int = pyvale.ErrorIntegrator([err_sys1,err_sys2],
                                          tc_array.get_measurement_shape())
    tc_array.set_pre_sys_err_integrator(sys_err_int)

    err_rand1 = pyvale.RandErrNormal(std=10.0)
    err_rand2 = pyvale.RandErrUniform(low=-10.0,high=10.0)
    rand_err_int = pyvale.ErrorIntegrator([err_rand1,err_rand2],
                                            tc_array.get_measurement_shape())
    tc_array.set_rand_err_integrator(rand_err_int)

    measurements = tc_array.get_measurements()

    print('\n'+80*'-')
    print('For a sensor: measurement = truth + sysematic error + random error')
    print(f'measurements.shape = {measurements.shape} = '+
          '(n_sensors,n_field_components,n_timesteps)\n')
    print("The truth, systematic error and random error arrays have the same "+
          "shape.")

    print(80*'-')
    print('Looking at the last 5 time steps (measurements) of sensor 0:')
    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')
    print("If we call the 'calc_measurements' method then the errors are "+
          "(re)calculated.")
    measurements = tc_array.calc_measurements()

    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))

    # We plot the first experiment for comparison
    (_,ax) = pyvale.plot_time_traces(tc_array,field_key)
    ax.set_title('Exp 1: called calc_measurements()')

    print(80*'-')
    print("If we call the 'get_measurements' method then the errors are the "+
          "same:")
    measurements = tc_array.get_measurements()

    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))

    # Plotting the second experiment to visulise the difference in errors
    (_,ax) = pyvale.plot_time_traces(tc_array,field_key)
    ax.set_title('Exp 2: called get_measurements()')

    print(80*'-')
    print("If we call the 'calc_measurements' method again we generate/sample"+
           "new errors:")
    measurements = tc_array.calc_measurements()

    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))

    # Plotting the second experiment to visulise the difference in errors
    (_,ax) = pyvale.plot_time_traces(tc_array,field_key)
    ax.set_title('Exp 3: called calc_measurements()')

    print(80*'-')

    plot_on = True
    if plot_on:
        plt.show()


if __name__ == '__main__':
    main()
