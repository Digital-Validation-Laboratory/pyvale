'''
================================================================================
example: displacement sensors on a 2d plate

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
    strain_field = pyvale.TensorField(sim_data,
                                    field_key,
                                    norm_components,
                                    dev_components,
                                    spat_dims)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    straingauge_array = pyvale.PointSensorArray(sens_pos,
                                                strain_field,
                                                None,
                                                descriptor,
                                                None,
                                                None)

    #---------------------------------------------------------------------------
    sg_array_norot = pyvale.PointSensorArray(sens_pos,
                                                strain_field,
                                                None,
                                                descriptor,
                                                None,
                                                None)
    rand_err_int = pyvale.ErrorIntegrator([pyvale.RandErrNormPercent(std_percent=5.0)],
                                        sg_array_norot.get_measurement_shape())
    sg_array_norot.set_rand_err_integrator(rand_err_int)

    meas_norot = sg_array_norot.get_measurements()

    #---------------------------------------------------------------------------
    angles = sens_pos.shape[0] * \
        (R.from_euler("zyx", [90, 0, 0], degrees=True),)

    sg_array_rot = pyvale.PointSensorArray(sens_pos,
                                                strain_field,
                                                None,
                                                descriptor,
                                                None,
                                                angles)

    rand_err_rot = pyvale.ErrorIntegrator([pyvale.RandErrNormPercent(std_percent=5.0)],
                                        sg_array_rot.get_measurement_shape())
    sg_array_rot.set_rand_err_integrator(rand_err_rot)

    meas_rot = sg_array_rot.get_measurements()

    #---------------------------------------------------------------------------
    plot_comp = 'strain_yy'
    pyvale.plot_time_traces(sg_array_norot,plot_comp)
    pyvale.plot_time_traces(sg_array_rot,plot_comp)
    plt.show()


if __name__ == "__main__":
    main()