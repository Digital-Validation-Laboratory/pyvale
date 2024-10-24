'''
================================================================================
example: strain gauges on a 2d plate

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
    #===========================================================================
    # Load Simulations as mooseherder.SimData objects
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
    # Creaet pyvale sensor arrays for thermal and mechanical data
    sim_data = sim_list[0]

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_field = 'temperature'
    tc_array = pyvale.SensorArrayFactory \
        .basic_thermocouple_array(sim_data,
                                  sens_pos,
                                  tc_field,
                                  spat_dims=2,
                                  sample_times=None,
                                  errs_pc=1.0)

    sg_field = 'strain'
    sg_array = pyvale.SensorArrayFactory \
        .basic_straingauge_array(sim_data,
                                  sens_pos,
                                  sg_field,
                                  spat_dims=2,
                                  sample_times=None,
                                  errs_pc=1.0)

    sensor_arrays = [tc_array,sg_array]

    #===========================================================================
    # Create and run the simulated experiment
    exp_sim = pyvale.ExperimentSimulator(sim_list,
                                        sensor_arrays,
                                        num_exp_per_sim=1000)

    exp_data = exp_sim.run_experiments()
    exp_stats = exp_sim.calc_stats()

    #===========================================================================
    print(80*"=")
    print("exp_data and exp_stats are lists where the index is the sensor array")
    print("position in the list as field components are not consistent dims.\n")

    print(80*"-")
    print("Thermal sensor array @ exp_data[0]")
    print(80*"-")
    print("shape=(n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)")
    print(f"{exp_data[0].shape=}")
    print()
    print("Stats are calculated over all experiments (axis=1)")
    print("shape=(n_sims,n_sensors,n_field_comps,n_time_steps)")
    print(f"{exp_stats[0].max.shape=}")
    print()
    print(80*"-")
    print("Mechanical sensor array @ exp_data[1]")
    print(80*"-")
    print("shape=(n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)")
    print(f"{exp_data[1].shape=}")
    print()
    print("shape=(n_sims,n_sensors,n_field_comps,n_time_steps)")
    print(f"{exp_stats[1].max.shape=}")
    print(80*"=")

    #===========================================================================
    # VISUALISE RESULTS
    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                      component="temperature",
                                      sens_array_num=0,
                                      sim_num=0)

    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                    component="strain_xx",
                                    sens_array_num=1,
                                    sim_num=2)
    plt.show()


if __name__ == "__main__":
    main()