'''
================================================================================
Analytic test case data - linear

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

""" TODO
An experiment should be able to:
- Perturb simulation input paramaters
    - Some might need to be sampled from a p-dist
    - Some as grid search
- Run simulation sweep

- Apply sensor array to each simulation combination
    - Generate 'N' experiments for each combination

- Plot error intervals for sensor traces

- Pickle the data for later use

An experiment has:
- A simulation / simulation workflow manager
    - The variables to perturb and how to perturb them
- A series of sensors of different types
- Simulation variables to analyse
- The number of experiments to run for each simulation
 """

@dataclass
class ExperimentConfig:
    num_exp_per_sim: int = 30
    variables: dict[str,(float,float)] | None = None

@dataclass
class ExperimentData:
    pass

class ExperimentSimulator:
    def __init__(self,
                 exp_config: ExperimentConfig,
                 sensor_arrays: list[pyvale.PointSensorArray]
                 ) -> None:
        self._exp_config = exp_config
        self._sensor_arrays = sensor_arrays
        self._exp_data = None

    def run_experiments(self) -> None:

        pass



def main() -> None:
    #===========================================================================
    # LOAD SIMULATION(S)
    base_path = Path("src/simcases/case18")
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
        .basic_thermocouple_array(sim_data,
                                  sens_pos,
                                  tc_field,
                                  spat_dims=2,
                                  sample_times=None,
                                  errs_pc=5.0)

    sg_field = 'strain'
    sg_array = pyvale.SensorArrayFactory \
        .basic_straingauge_array(sim_data,
                                  sens_pos,
                                  sg_field,
                                  spat_dims=2,
                                  sample_times=None,
                                  errs_pc=5.0)

    sensor_arrays = [tc_array,sg_array]

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n')

    #===========================================================================
    # CREATE & RUN THE SIMULATED EXPERIMENT
    num_exp_per_sim = 1000
    # shape=(n_arrays,...,n_sims,n_exps,n_sens,n_comps,n_time_steps)
    n_arrays = len(sensor_arrays)
    n_sims = len(sim_list)
    exp_data = [None]*n_arrays

    for ii,aa in enumerate(sensor_arrays):
        meas_array = np.ones((n_sims,num_exp_per_sim)+aa.get_measurement_shape())

        for jj,ss in enumerate(sim_list):
            aa.field.set_sim_data(ss)

            for ee in range(num_exp_per_sim):
                meas_array[jj,ee,:,:,:] = aa.calc_measurements()

        exp_data[ii] = meas_array

    #===========================================================================
    # VISUALISE RESULTS
    component = 'temperature'
    sens_array_num = 0
    sens_num = -1
    sim_num = 0

    descriptor = sensor_arrays[0].descriptor
    comp_ind = sensor_arrays[0].field.get_component_index(component)
    samp_time = sensor_arrays[0].get_sample_times()

    plot_opts = pyvale.GeneralPlotOpts()
    trace_opts = pyvale.SensorTraceOpts()

    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    for ee in range(num_exp_per_sim):
        ax.plot(samp_time,
                exp_data[sens_array_num][sim_num,ee,sens_num,comp_ind,:],
                "o",
                lw=plot_opts.lw,
                ms=plot_opts.ms,
                color=plot_opts.colors[0 % plot_opts.n_colors])

    ax.set_xlabel(trace_opts.time_label,
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(descriptor.create_label(comp_ind),
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    if trace_opts.time_min_max is None:
        ax.set_xlim((np.min(samp_time),np.max(samp_time))) # type: ignore
    else:
        ax.set_xlim(trace_opts.time_min_max)

    trace_opts.legend = False
    if trace_opts.legend:
        ax.legend(prop={"size":plot_opts.font_leg_size},loc='best')

    plt.grid(True)
    plt.draw()

    plt.show()




if __name__ == "__main__":
    main()