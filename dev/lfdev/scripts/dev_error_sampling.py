'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale


def main() -> None:
    """pyvale example:
    """
    data_path = Path('src/data/case13_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'temperature'    # Same as in the moose input and SimData node_var key
    t_field = pyvale.FieldScalar(sim_data,field_name,spat_dims)

    n_sens = (3,1,1)    # Number of sensor (x,y,z)
    x_lims = (0.0,2.0)  # Limits for each coord in sim length units
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.SensorArrayPoint(sens_pos,t_field)

    rand_err1 = pyvale.ErrRandNormPercent(std_percent=10.0)
    rand_err_int = pyvale.ErrIntegrator([rand_err1],
                                            tc_array.get_measurement_shape())
    tc_array.set_random_err_integrator(rand_err_int)


    start_time = time.perf_counter()

    n_samples = int(1e5)

    m_shape = tc_array.get_measurement_shape()
    measurements = np.zeros((n_samples,
                             m_shape[0],
                             m_shape[1],
                             m_shape[2]))
    for nn in range(n_samples):
        measurements[nn,:,:,:] = tc_array.calc_measurements()

    end_time = time.perf_counter()

    pyvale.plot_time_traces(tc_array,field_name)

    print("\n"+80*"=")
    print(f"Elapsed time: {end_time-start_time} s")
    print(80*"="+"\n")

    n_bins = 40
    _, axs = plt.subplots(1,3,tight_layout=True)
    axs[0].hist(np.squeeze(measurements[:,0,0,-1]),bins=n_bins)
    axs[1].hist(np.squeeze(measurements[:,1,0,-1]),bins=n_bins)
    axs[2].hist(np.squeeze(measurements[:,2,0,-1]),bins=n_bins)

    plt.show()


if __name__ == '__main__':
    main()
