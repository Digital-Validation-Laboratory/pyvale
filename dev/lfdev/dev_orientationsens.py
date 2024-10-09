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

    descriptor = pyvale.SensorDescriptorFactory.displacement_descriptor()

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyvale.VectorField(sim_data,field_key,components,spat_dims)

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    orientations = sens_pos.shape[0] * \
        (R.from_euler("zyx", [45, 0, 0], degrees=True),)

    print(f"{R.from_euler("zyx", [45, 0, 0], degrees=True).as_matrix()=}")

    disp_sens_array = pyvale.PointSensorArray(sens_pos,
                                              disp_field,
                                              None,
                                              descriptor,
                                              None,
                                              orientations)

    measurements = disp_sens_array.get_measurements()

    print(80*"=")
    print()
    print(f"{measurements.shape=}")
    meas_shape = measurements.shape

    # Need to add the third component to do the rotation
    measurements = np.concatenate(
        (measurements,np.zeros((meas_shape[0],1,meas_shape[2]))),
        axis=1)


    print(f"{measurements.shape=}")
    # NOTE:)
    # Rotation = object rotates coords fixed, sin neg row 1
    # Transformation = coords rotate object fixed, win neg row 2, transpose scipy
    rot = R.from_euler("zyx", [90, 0, 0], degrees=True)
    rmat = rot.as_matrix().T

    sens1 = np.squeeze(measurements[0,:,:])
    print()
    print(f"{sens1.shape=}")
    rmat_x_sens1 = np.matmul(rmat,sens1)

    npaa = np.apply_along_axis(np.matmul,0,rmat.T,measurements)
    print(f"{npaa.shape=}")
    print()
    check_npaa = np.squeeze(npaa[0,:,:]).T
    print(f"{rmat_x_sens1.shape=}")
    print(f"{check_npaa.shape=}")
    print(f"{np.allclose(rmat_x_sens1,check_npaa)=}")

    print(80*"=")

    #pyvale.plot_time_traces(disp_sens_array,'disp_x')
    #pyvale.plot_time_traces(disp_sens_array,'disp_y')
    #plt.show()


if __name__ == "__main__":
    main()