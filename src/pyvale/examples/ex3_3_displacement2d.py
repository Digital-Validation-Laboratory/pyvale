'''
================================================================================
Example: displacement sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale

def main() -> None:
    """pyvale example: displacement sensors on a 2D plate with a hole
    ----------------------------------------------------------------------------
    - Demonstrates sensor rotation for vector fields
    """
    data_path = pyvale.DataSet.mechanical_2d_output_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptorFactory.displacement_descriptor()

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyvale.FieldVector(sim_data,field_key,components,spat_dims)

    n_sens = (2,2,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sensor_positions = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data_norot = pyvale.SensorData(positions=sensor_positions,
                                        sample_times=sample_times)

    disp_sens_norot = pyvale.SensorArrayPoint(sens_data_norot,
                                              disp_field,
                                              descriptor)

    meas_norot = disp_sens_norot.get_measurements()

    sens_angles = sensor_positions.shape[0] * \
        (Rotation.from_euler("zyx", [45, 0, 0], degrees=True),)

    sens_data_rot = pyvale.SensorData(positions=sensor_positions,
                                      sample_times=sample_times,
                                      angles=sens_angles)

    disp_sens_rot = pyvale.SensorArrayPoint(sens_data_rot,
                                            disp_field,
                                            descriptor)


    angle_offset = np.zeros_like(sensor_positions)
    angle_offset[:,0] = 1.0 # only rotate about z in 2D
    angle_error_data = pyvale.ErrFieldData(ang_offset_zyx=angle_offset)

    sys_err_rot = pyvale.ErrSysField(disp_field,angle_error_data)

    sys_err_int = pyvale.ErrIntegrator([sys_err_rot],
                                         sens_data_rot,
                                         disp_sens_rot.get_measurement_shape())
    disp_sens_rot.set_error_integrator(sys_err_int)

    meas_rot = disp_sens_rot.get_measurements()


    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyvale.print_measurements(disp_sens_rot,
                              (sens_num-1,sens_num),
                              (0,1),
                              (meas_rot.shape[2]-5,meas_rot.shape[2]))
    print(80*'-')

    plot_field = 'disp_x'
    if plot_field == 'disp_x':
        pv_plot = pyvale.plot_point_sensors_on_sim(disp_sens_rot,'disp_x')
        pv_plot.show(cpos="xy")
    elif plot_field == 'disp_y':
        pv_plot = pyvale.plot_point_sensors_on_sim(disp_sens_rot,'disp_y')
        pv_plot.show(cpos="xy")

    pyvale.plot_time_traces(disp_sens_norot,plot_field)
    pyvale.plot_time_traces(disp_sens_rot,plot_field)
    plt.show()


if __name__ == "__main__":
    main()