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

    pre_sys_err1 = pyvale.SysErrUniform(low=-20.0,high=20.0)
    pre_sys_err2 = pyvale.SysErrNormal(std=20.0)
    sys_err_int = pyvale.ErrorIntegrator([pre_sys_err1,pre_sys_err2],
                                          tc_array.get_measurement_shape())
    tc_array.set_pre_sys_err_integrator(sys_err_int)

    rand_err1 = pyvale.RandErrNormal(std=10.0)
    rand_err2 = pyvale.RandErrUniform(low=-10.0,high=10.0)
    rand_err_int = pyvale.ErrorIntegrator([rand_err1,rand_err2],
                                            tc_array.get_measurement_shape())
    tc_array.set_rand_err_integrator(rand_err_int)

    measurements = tc_array.get_measurements()



if __name__ == '__main__':
    main()
