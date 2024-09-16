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
    data_path = Path('src/data/case16_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    #===========================================================================
    # CREATE SENSOR ARRAYS
    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_field = 'temperature'
    tc_array = pyvale.SensorArrayFactory \
        .basic_thermocouple_array(sim_data,
                                  sens_pos,
                                  tc_field,
                                  spat_dims=3,
                                  sample_times=None,
                                  errs_pc=5.0)

    sg_field = 'strain'
    sg_array = pyvale.SensorArrayFactory \
        .basic_straingauge_array(sim_data,
                                  sens_pos,
                                  sg_field,
                                  spat_dims=3,
                                  sample_times=None,
                                  errs_pc=5.0)

    sensor_arrays = [tc_array,sg_array]

    #===========================================================================
    # CREATE SIMULATED EXPERIMENT
    num_exp_per_sim = 30
    for ss in 
    for ee in range(num_exp_per_sim):


    #===========================================================================
    # RUN SIMULATED EXPERIMENT

    #===========================================================================
    # VISUALISE RESULTS


if __name__ == "__main__":
    main()