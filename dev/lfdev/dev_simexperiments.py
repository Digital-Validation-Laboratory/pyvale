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


@dataclass
class ExperimentConfig:
    num_exp_per_sim: int = 30
    variables: dict[str,(float,float)] | None = None

@dataclass
class ExperimentData:
    exp_config: ExperimentConfig | None = None
    exp_data: list[np.ndarray] | None  = None
    exp_stats: list[np.ndarray] | None = None

@dataclass
class ExperimentStats:
    avg: np.ndarray | None = None
    std: np.ndarray | None = None
    cov: np.ndarray | None = None
    max: np.ndarray | None = None
    min: np.ndarray | None = None
    med: np.ndarray | None = None
    q25: np.ndarray | None = None
    q75: np.ndarray | None = None
    mad: np.ndarray | None = None

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

    def calc_stats(self) -> None:
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

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for last sensor:\n{measurements[-1,0,:]}\n')

    #===========================================================================
    # CREATE & RUN THE SIMULATED EXPERIMENT
    num_exp_per_sim = 1000
    n_arrays = len(sensor_arrays)
    n_sims = len(sim_list)
    # shape=list[n_arrays](n_sims,n_exps,n_sens,n_comps,n_time_steps)
    exp_data = [None]*n_arrays

    for ii,aa in enumerate(sensor_arrays):
        meas_array = np.zeros((n_sims,num_exp_per_sim)+
                               aa.get_measurement_shape())

        for jj,ss in enumerate(sim_list):
            aa.field.set_sim_data(ss)

            for ee in range(num_exp_per_sim):
                meas_array[jj,ee,:,:,:] = aa.calc_measurements()

        exp_data[ii] = meas_array

    #===========================================================================
    # ANALYSE EXPERIMENTAL DATA
    # Fix Sim, Fix Sensor, Stats over Exp

    # shape=list[n_arrays](n_sims,n_exps,n_sens,n_comps,n_time_steps)
    exp_stats = [None]*n_arrays
    for ii,aa in enumerate(sensor_arrays):
        array_stats = ExperimentStats()
        array_stats.max = np.max(exp_data[ii],axis=1)
        array_stats.min = np.min(exp_data[ii],axis=1)
        array_stats.avg = np.mean(exp_data[ii],axis=1)
        array_stats.std = np.std(exp_data[ii],axis=1)
        array_stats.med = np.median(exp_data[ii],axis=1)
        array_stats.q25 = np.quantile(exp_data[ii],0.25,axis=1)
        array_stats.q75 = np.quantile(exp_data[ii],0.75,axis=1)
        array_stats.mad = np.median(np.abs(exp_data[ii] - np.median(exp_data[ii],axis=1,keepdims=True)),axis=1)
        exp_stats[ii] = array_stats

    print(80*"=")
    print(f"{exp_data[0].shape=}")
    print(f"{exp_stats[0].max.shape=}")
    print(f"{exp_data[1].shape=}")
    print(f"{exp_stats[1].max.shape=}")
    print(exp_stats[0].max[0,-1,0,:])
    print(80*"=")

    #===========================================================================
    # VISUALISE RESULTS
    component = 'temperature'
    sens_array_num = 0
    sens_to_plot = None
    sim_num = 0
    plot_all_exp_points = True

    descriptor = sensor_arrays[sens_array_num].descriptor
    comp_ind = sensor_arrays[sens_array_num].field.get_component_index(component)
    samp_time = sensor_arrays[sens_array_num].get_sample_times()
    num_sens = sensor_arrays[sens_array_num].get_measurement_shape()[0]

    if sens_to_plot is None:
        sens_to_plot = range(num_sens)

    plot_opts = pyvale.GeneralPlotOpts()
    trace_opts = pyvale.SensorTraceOpts()

    fig, ax = plt.subplots(figsize=plot_opts.single_fig_size,
                           layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    if plot_all_exp_points:
        for ss in sens_to_plot:
            for ee in range(num_exp_per_sim):
                ax.plot(samp_time,
                        exp_data[sens_array_num][sim_num,ee,ss,comp_ind,:],
                        "o",
                        lw=plot_opts.lw,
                        ms=plot_opts.ms,
                        color=plot_opts.colors[ss % plot_opts.n_colors])

    for ss in sens_to_plot:
        ax.plot(samp_time,
                exp_stats[sens_array_num].avg[sim_num,ss,comp_ind,:],
                "-",
                lw=plot_opts.lw,
                ms=plot_opts.ms,
                color=plot_opts.colors[ss % plot_opts.n_colors])
        ax.fill_between(samp_time,
                exp_stats[sens_array_num].min[sim_num,ss,comp_ind,:],
                exp_stats[sens_array_num].max[sim_num,ss,comp_ind,:],
                color=plot_opts.colors[ss % plot_opts.n_colors],
                alpha=0.2)


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