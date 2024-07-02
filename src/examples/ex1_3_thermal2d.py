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
    """pyvale example: specifying a sensor sampling time and controlling plots.
    """
    # Use mooseherder to read the exodus and get a SimData object
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    # The SensorDescriptor holds strings used to label plots and visualisations
    descriptor = pyvale.SensorDescriptor()
    descriptor.name = 'Temperature'
    descriptor.symbol = 'T'
    descriptor.units = r'^{\circ}C'
    descriptor.tag = 'TC'

    # Create a Field object that will allow the sensors to interpolate the sim
    # data field of interest quickly by using the mesh and shape functions
    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_key = 'temperature'    # Same as in the moose input and SimData node_var key
    t_field = pyvale.ScalarField(sim_data,field_key,spat_dims)

    # This creates a grid of 3x2 sensors in the xy plane
    n_sens = (5,2,1)    # Number of sensor (x,y,z)
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
    tc_array.set_pre_sys_err_integrator(sys_err_int)

    # Random errors are also integrated and we can chain objects to calculate
    # multiple random error functions. The random error is sampled repeatedly
    # for each time step.
    err_rand1 = pyvale.RandErrNormal(std=10.0)
    err_rand2 = pyvale.RandErrUniform(low=-10.0,high=10.0)
    rand_err_int = pyvale.ErrorIntegrator([err_rand1,err_rand2],
                                            tc_array.get_measurement_shape())
    tc_array.set_rand_err_integrator(rand_err_int)

    # We can get an array of measurements from the sensor array for all time
    # steps in the simulation. A measurement is calculated as follows:
    # measurement = truth + systematic_error + random_error
    measurements = tc_array.get_measurements()



if __name__ == '__main__':
    main()
