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
    data_path = Path('data/examplesims/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    spat_dims = 2
    field_name = 'temperature'
    t_field = pyvale.ScalarField(sim_data,field_name,spat_dims)

    n_sens = (3,2,1)
    x_lims = (0.0,2.0)
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.PointSensorArray(sens_pos,t_field)

    err_sys1 = pyvale.SysErrUniform(low=-20.0,high=20.0)
    err_sys2 = pyvale.SysErrNormal(std=20.0)
    sys_err_int = pyvale.ErrorIntegrator([err_sys1,err_sys2],
                                          tc_array.get_measurement_shape())
    tc_array.set_sys_err_integrator(sys_err_int)

    measurements = tc_array.calc_measurements()

    print(80*'-')
    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))
    print(80*'-')
    pyvale.print_measurements(tc_array,
                              (0,1),
                              (0,1),
                              (measurements.shape[2]-5,measurements.shape[2]))

    (_,ax) = pyvale.plot_time_traces(tc_array,field_name)
    print(80*'-')

    #plt.show()


if __name__ == '__main__':
    main()
