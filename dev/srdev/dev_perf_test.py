'''
================================================================================
pycave: the python computer aided validation engine

License: LGPL-2.1
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

import time
from pprint import pprint
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import mooseherder as mh
import pycave


N_SENS = (2**4,2**5,2**6,2**8,2**10)
N_SAMPLES = (1e1,1e2,1e3)
N_REPEATS = 20


def main() -> None:
    data_path = Path('data/plate_2d_thermal_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    spat_dims = 2       # Specify that we only have 2 spatial dimensions
    field_name = 'temperature'    # Same as in the moose input and SimData node_var key
    t_field = pycave.Field(sim_data,field_name,spat_dims)

    x_lims = (0.0,2.0)  # Limits for each coord in sim length units
    y_lims = (0.0,1.0)
    z_lims = (0.0,0.0)

    perf_array = np.zeros([len(N_SENS),len(N_SAMPLES)])

    for ii,ss in enumerate(N_SENS):
            for jj,nn in enumerate(N_SAMPLES):
                total_time = 0.0

                for _ in range(N_REPEATS):
                    start_time = time.perf_counter()

                    n_sens = (ss,ss,1)    # Number of sensor (x,y,z)
                    sens_pos = pycave.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

                    sample_times = np.arange(t_field.get_time_steps()[0],
                                            t_field.get_time_steps()[1],
                                            nn)

                    tc_array = pycave.ThermocoupleArray(sens_pos,t_field,sample_times)

                    tc_array.set_uniform_systematic_err_func(low=-10.0,high=10.0)
                    tc_array.set_normal_random_err_func(std_dev=5.0)

                    measurements = tc_array.get_measurements()

                    process_time = time.perf_counter() - start_time

                    total_time += process_time

                avg_process_time = total_time / N_REPEATS
                perf_array[ii,jj] = avg_process_time

    perf_table = np.zeros([len(N_SENS)+1,len(N_SAMPLES)+1])
    perf_table[1:,1:] = perf_array
    perf_table[1:,0] = np.array(N_SENS)
    perf_table[0,1:] = np.array(N_SAMPLES)

    np.set_printoptions(linewidth=np.inf) # type: ignore
    print()
    print(80*'=')
    print('Performance time =')
    pprint(perf_table)
    print(80*'=')
    print()

    return





    trace_plot_mode = 'off'

    (fig,ax) = tc_array.plot_time_traces(plot_truth=False,plot_sim=True)
    if trace_plot_mode == 'interactive':
        #ax.set_xlim([0.0,5.0])
        plt.show()
    if trace_plot_mode == 'save_fig':
        save_traces = Path('examples/images/plate_thermal_2d_traces.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
