'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    data_path = Path('src/data/case13_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    spat_dims = 2
    field_name = 'temperature'
    t_field = pyvale.ScalarField(sim_data,field_name,spat_dims)

    n_sens = (3,1,1)
    x_lims = (0.0,2.0)
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.PointSensorArray(sens_pos,t_field)

    field_sys_err1 = pyvale.SysErrPointPosition(t_field,
                                           sens_pos,
                                           (0.1,0.1,None))

    spatial_int = pyvale.QuadratureFactory.quad_2d_9points(t_field,
                                                        sens_pos,
                                                        np.array((0.1,0.1,0.0)))

    field_sys_err2 = pyvale.SysErrStaticSpatialAverage(t_field,
                                                 spatial_int)

    field_sys_err3 = pyvale.SysErrDynamicSpatialAverage(t_field,
                                                 spatial_int,
                                                 sens_pos,
                                                 (0.1,0.1,None))

    pre_syserr_int = pyvale.ErrorIntegrator([field_sys_err2],
                                            tc_array.get_measurement_shape())
    tc_array.set_indep_sys_err_integrator(pre_syserr_int)

    measurements = tc_array.calc_measurements()

    print_meas = True
    if print_meas:
        print(80*'-')
        pyvale.print_measurements(tc_array,
                                (measurements.shape[0]-1,measurements.shape[0]),
                                (0,1),
                                (measurements.shape[2]-10,measurements.shape[2]))
        print(80*'-')


        #pyvale.plot_time_traces(tc_array,field_name)
        #plt.show()


if __name__ == '__main__':
    main()
