'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvale
import mooseherder as mh


def main() -> None:
    #===========================================================================
    # LOAD SIMULATION(S)
    base_path = Path("src/data")
    data_paths = [base_path / 'case18_1_out.e',
                  base_path / 'case18_2_out.e',
                  base_path / 'case18_3_out.e']

    sim_list = []
    for pp in data_paths:
        sim_data = mh.ExodusReader(pp).read_all_sim_data()
        # Scale to mm to make 3D visualisation scaling easier
        sim_data.coords = sim_data.coords*1000.0 # type: ignore
        sim_list.append(sim_data)

    #===========================================================================
    # CREATE SENSOR ARRAYS
    sim_data = sim_list[0]
    print(sim_data.node_vars["temperature"])
    print(sim_data.coords)

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_field = 'temperature'
    tc_array = pyvale.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_pos,
                                  tc_field,
                                  spat_dims=2,
                                  sample_times=None,
                                  errs_pc=1.0)

    sg_field = 'strain'
    sg_array = pyvale.SensorArrayFactory \
        .strain_gauges_basic_errs(sim_data,
                                  sens_pos,
                                  sg_field,
                                  spat_dims=2,
                                  sample_times=None,
                                  errs_pc=1.0)

    sensor_arrays = [tc_array,sg_array]

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n')

    #===========================================================================
    # CREATE & RUN THE SIMULATED EXPERIMENT
    exp_sim = pyvale.ExperimentSimulator(sim_list,
                                  sensor_arrays,
                                  num_exp_per_sim=1000)

    exp_data = exp_sim.run_experiments()
    exp_stats = exp_sim.calc_stats()


    #===========================================================================
    # ANALYSE EXPERIMENTAL DATA
    # Fix Sim, Fix Sensor, Stats over Exp

    print(80*"=")
    print(f"{exp_data[0].shape=}")
    print(f"{exp_stats[0].max.shape=}")
    print(f"{exp_data[1].shape=}")
    print(f"{exp_stats[1].max.shape=}")
    print(exp_stats[0].max[0,-1,0,:])
    print(80*"=")

    #===========================================================================
    # VISUALISE RESULTS
    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                      component="temperature",
                                      sens_array_num=0,
                                      sim_num=0)
    plt.show()


if __name__ == "__main__":
    main()