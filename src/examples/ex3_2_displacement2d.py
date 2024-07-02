'''
================================================================================
example: displacement sensors on a 2d plate

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
    data_path = Path('simcases/case17/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptor()
    descriptor.name = 'Displacement'
    descriptor.symbol = r'u'
    descriptor.units = r'm'
    descriptor.tag = 'DS'
    descriptor.components = ('x','y','z')

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyvale.VectorField(sim_data,field_key,components,spat_dims)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    disp_sens_array = pyvale.PointSensorArray(sens_pos,
                                              disp_field,
                                              None,
                                              descriptor)

    err_sys1 = pyvale.SysErrUniform(low=-0.01e-3,high=0.01e-3)
    sys_err_int = pyvale.ErrorIntegrator([err_sys1],
                                          disp_sens_array.get_measurement_shape())
    disp_sens_array.set_pre_sys_err_integrator(sys_err_int)

    err_rand1 = pyvale.RandErrNormal(std=0.01e-3)
    rand_err_int = pyvale.ErrorIntegrator([err_rand1],
                                            disp_sens_array.get_measurement_shape())
    disp_sens_array.set_rand_err_integrator(rand_err_int)

    plot_field = 'disp_x'
    if plot_field == 'disp_x':
        pv_plot = pyvale.plot_sensors_on_sim(disp_sens_array,'disp_x')
        pv_plot.show()
    elif plot_field == 'disp_y':
        pv_plot = pyvale.plot_sensors_on_sim(disp_sens_array,'disp_y')
        pv_plot.show()

    pyvale.plot_time_traces(disp_sens_array,'disp_x')
    pyvale.plot_time_traces(disp_sens_array,'disp_y')
    plt.show()


if __name__ == "__main__":
    main()