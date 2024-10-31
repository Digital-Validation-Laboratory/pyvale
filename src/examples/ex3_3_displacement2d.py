'''
================================================================================
Example: displacement sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import mooseherder as mh
import pyvale

def main() -> None:
    """pyvale example: rotation of vector field sensors
    """
    data_path = Path('src/data/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptorFactory.displacement_descriptor()

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyvale.VectorField(sim_data,field_key,components,spat_dims)

    n_sens = (2,2,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = False
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data_norot = pyvale.SensorData(positions=sens_pos,
                                        sample_times=sample_times)

    disp_sens_norot = pyvale.PointSensorArray(sens_data_norot,
                                              disp_field,
                                              descriptor)

    meas_norot = disp_sens_norot.get_measurements()

    sens_angles = sens_pos.shape[0] * \
        (Rotation.from_euler("zyx", [45, 0, 0], degrees=True),)

    sens_data_rot = pyvale.SensorData(positions=sens_pos,
                                      sample_times=sample_times,
                                      angles=sens_angles)

    disp_sens_rot = pyvale.PointSensorArray(sens_data_rot,
                                            disp_field,
                                            descriptor)

    offset_angles = np.array([1,0,0]) # degrees
    sys_err_rot = pyvale.SysErrAngleOffset(disp_field,
                                           sens_pos,
                                           sens_angles,
                                           offset_angles,
                                           sample_times)

    sys_err_int = pyvale.ErrorIntegrator([sys_err_rot],
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

    pyvale.plot_time_traces(disp_sens_norot,'disp_x')
    pyvale.plot_time_traces(disp_sens_norot,'disp_y')

    pyvale.plot_time_traces(disp_sens_rot,'disp_x')
    pyvale.plot_time_traces(disp_sens_rot,'disp_y')
    plt.show()


if __name__ == "__main__":
    main()