'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pprint import pprint
from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt
from mooseherder import (MooseHerd,
                         MooseConfig,
                         MooseRunner,
                         GmshRunner,
                         InputModifier,
                         DirectoryManager,
                         ExodusReader)
import pyvale
import time,os
import numpy as np
import pandas as pd

#======================================
# Change this to run a different case
CASE_STR = 'case01_2d_thermal_steady'
OUT_STR = 'case01_sim_err_only'
#======================================

CASE_FILES = (CASE_STR+'.geo',CASE_STR+'.i')
CASE_DIR = Path('examples/'+CASE_STR+'/')

USER_DIR = Path.home() / 'git/herding-moose/'
OUTPUT_DIR = USER_DIR / f'pyvale/examples/{OUT_STR}/'
if not Path.exists(OUTPUT_DIR):
    OUTPUT_DIR.mkdir()

FORCE_GMSH = False

NUM_RUNS=100

def print_attrs(in_obj: Any) -> None:
    _ = [print(aa) for aa in dir(in_obj) if '__' not in aa]

def main() -> None:

    run_moose = True
    if run_moose:
        read_results = True
        save_vals = True
    else:
        read_results = False
        save_vals = False
    
    
    # edit moose file
    moose_input = Path('examples/' + CASE_STR + '/' + CASE_STR + '.i')
    config_path = Path.cwd() / '../mooseherder/moose-config.json'
    moose_config = MooseConfig().read_config(config_path)
    moose_modifier = InputModifier(moose_input,'#','')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 2,
                          redirect_out = True)

    dir_manager = DirectoryManager(n_dirs=1)
    
    # Start the herd and create working directories
    herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)

    # Set the parallelisation options, we have 8 combinations of variables and
    # 4 MOOSE intances running, so 2 runs will be saved in each working directory
    herd.set_num_para_sims(n_para=1)

     # Send all the output to the examples directory and clear out old output
    dir_manager.set_base_dir(OUTPUT_DIR)
    dir_manager.clear_dirs()
    dir_manager.create_dirs()
    
    # Create variables to sweep
    
    numRandPars = 1000
    stdDevPerc = 0.1 # 10% std dev in material parameters
    
    cuDensity = np.random.normal(loc=8829.0, scale=stdDevPerc*8829.0, size=numRandPars)
    cuThermCond = np.random.normal(loc=384.0, scale=stdDevPerc*384.0, size=numRandPars)
    cuSpecHeat = np.random.normal(loc=406.0, scale=stdDevPerc*406.0, size=numRandPars)
    
    moose_vars = list([])
    var_array = list([])
    for dd in range(len(cuDensity)):
        moose_vars.append([{'cuDensity':cuDensity[dd], 'cuThermCond':cuThermCond[dd], 'cuSpecHeat':cuSpecHeat[dd]}])
        var_array.append([cuDensity[dd],cuThermCond[dd],cuSpecHeat[dd]])
        
    var_array = np.array(var_array)
    
    print(f"Variable array shape = {var_array.shape}")
    columns = moose_vars[0][0].keys()
    var_df = pd.DataFrame(var_array,columns=columns)
    if save_vals:
        var_path = OUTPUT_DIR / 'moose_vars.csv'
        var_df.to_csv(var_path,columns=columns,index=False)
    
    print('Herd sweep variables:')
    for vv in moose_vars:
        print(vv)
        
    if run_moose:
        
        print("-"*80)
        print('EXAMPLE: Run MOOSE in sequence')
        print("-"*80)
        
        herd.run_sequential(moose_vars)

        print(f'Run time (seq) = {herd.get_sweep_time():.3f} seconds')
        print("-"*80)
        print()
    
    
    if read_results:
        # Get clean results
        for nn in range(len(moose_vars)):
    
            # Use mooseherder to read the exodus and get a SimData object
            data_path = OUTPUT_DIR / f'sim-workdir-1/sim-1-{nn+1}_out.e'
            data_reader = ExodusReader(data_path)
            sim_data = data_reader.read_all_sim_data()

            #  Create a Field object that will allow the sensors to interpolate the sim
            # data field of interest quickly by using the mesh and shape functions
            spat_dims = 2       # Specify that we only have 2 spatial dimensions
            field_name = 'temperature'    # Same as in the moose input and SimData node_var key
            t_field = pyvale.Field(sim_data,field_name,spat_dims)

            # This creates a grid of 3x2 sensors in the xy plane
            n_sens = (3,2,1)    # Number of sensor (x,y,z)
            x_lims = (0.0,100.0e-03)#(0.0,2.0)  # Limits for each coord in sim length units
            y_lims = (0.0,50.0e-03)#(0.0,1.0)
            z_lims = (0.0,0.0)#(0.0,0.0)
            # Gives a n_sensx3 array of sensor positions where each row is a sensor 
            # with coords (x,y,z) - can also just manually create this array
            sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

            # Now we create a thermocouple array with with the sensor positions and the
            # temperature field from the simulation
            tc_array = pyvale.ThermocoupleArray(sens_pos,t_field)

            measurements = tc_array.get_measurements()

            # Now save the desired data into a csv
            
            if save_vals:
                read_config = data_reader.get_read_config()
                sim_data = data_reader.read_sim_data(read_config)
        
                # save sensor positions
                columns = [f"s{i+1}" for i in range(sens_pos.shape[0])]
                sens_pos_df = pd.DataFrame(np.transpose(sens_pos),columns=columns)
                pos_path = OUTPUT_DIR / f'sens_pos_{nn+1:03}.csv'
                sens_pos_df.to_csv(pos_path,columns=columns,index=False)
            
                # save sensor data
                sens_data = np.concatenate([np.reshape(sim_data.time,(1,len(sim_data.time))),measurements])
                print(sens_data.shape)
                columns = ["time"]
                for i in range(measurements.shape[0]):
                    columns.append(f"s{i+1}")
                sens_data_df = pd.DataFrame(np.transpose(sens_data),columns=columns)
                data_path = OUTPUT_DIR / f'sim_data_{nn+1:03}.csv'
                sens_data_df.to_csv(data_path,columns=columns,index=False)


if __name__ == '__main__':
    main()
