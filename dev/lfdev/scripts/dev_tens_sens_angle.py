'''
================================================================================
example: strain sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import mooseherder as mh
import pyvale

def main() -> None:
    """pyvale example:
    """
    data_path = Path('src/data/case17_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    descriptor = pyvale.SensorDescriptorFactory.strain_descriptor()

    spat_dims = 2
    field_key = 'strain'
    norm_components = ('strain_xx','strain_yy')
    dev_components = ('strain_xy',)
    strain_field = pyvale.FieldTensor(sim_data,
                                    field_key,
                                    norm_components,
                                    dev_components,
                                    spat_dims)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    straingauge_array = pyvale.SensorArrayPoint(sens_pos,
                                                strain_field,
                                                None,
                                                descriptor,
                                                None,
                                                None)

    #---------------------------------------------------------------------------
    sg_array_norot = pyvale.SensorArrayPoint(sens_pos,
                                                strain_field,
                                                None,
                                                descriptor,
                                                None,
                                                None)

    meas_norot = sg_array_norot.get_measurements()

    #---------------------------------------------------------------------------
    sens_angles = sens_pos.shape[0] * \
        (R.from_euler("zyx", [0, 0, 0], degrees=True),)

    sg_array_rot = pyvale.SensorArrayPoint(sens_pos,
                                                strain_field,
                                                None,
                                                descriptor,
                                                None,
                                                sens_angles)

    offset_angles = np.array([1,0,0])
    sys_err_rot = pyvale.ErrSysAngleOffset(strain_field,
                                     sens_pos,
                                     sens_angles,
                                     offset_angles,
                                     None)
    sys_err_int = pyvale.ErrIntegrator([sys_err_rot],
                                        sg_array_rot.get_measurement_shape())
    sg_array_rot.set_systematic_err_integrator_independent(sys_err_int)


    meas_rot = sg_array_rot.get_measurements()

    #---------------------------------------------------------------------------
    print(80*'-')
    sens_num = 4
    print('The last 5 time steps (measurements) of sensor {sens_num}:')
    pyvale.print_measurements(sg_array_rot,
                              (sens_num-1,sens_num),
                              (1,2),
                              (meas_rot.shape[2]-5,meas_rot.shape[2]))
    print(80*'-')

    plot_comp = 'strain_yy'
    pyvale.plot_time_traces(sg_array_norot,plot_comp)
    pyvale.plot_time_traces(sg_array_rot,plot_comp)
    plt.show()


if __name__ == "__main__":
    main()