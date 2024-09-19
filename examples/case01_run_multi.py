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
import time
import numpy as np
import pandas as pd

#======================================
# Change this to run a different case
CASE_STR = 'case01_2d_thermal_steady'
#======================================

CASE_FILES = (CASE_STR+'.geo',CASE_STR+'.i')
CASE_DIR = Path('examples/'+CASE_STR+'/')

USER_DIR = Path.home() / 'git/herding-moose/'
OUTPUT_DIR = USER_DIR / 'pyvale/examples/case01_out/'

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
    

    # set up uncertainty on model parameters
    
    
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
    dir_manager.set_base_dir(Path('examples/'))
    dir_manager.clear_dirs()
    dir_manager.create_dirs()
    
    # Create variables to sweep in a list of dictionaries, 125 combinations possible.
    #cuDensity = [8829.0,8834.0,8824.0]#[8.91e+03,8.92e+03,8.93e+03,8.96e+03,8.90e+03]
    #cuThermCond = [384.0,379.0,389.0]#[401,398,393,403,395] #W/m/k # 398 W/(m·K) with uncertainty of 5 W/(m·K)
    #cuSpecHeat = [406.0,401.0,411.0]#[0.386,0.376,0.380,0.390,0.385]
    
    numRandPars = 1000
    stdDevPerc = 0.1
    
    cuDensity = np.random.normal(loc=8829.0, scale=stdDevPerc*8829.0, size=numRandPars)
    cuThermCond = np.random.normal(loc=384.0, scale=stdDevPerc*384.0, size=numRandPars)
    cuSpecHeat = np.random.normal(loc=406.0, scale=stdDevPerc*406.0, size=numRandPars)
    
    moose_vars = list([])
    var_array = list([])
    for dd in range(len(cuDensity)):
        moose_vars.append([{'cuDensity':cuDensity[dd], 'cuThermCond':cuThermCond[dd], 'cuSpecHeat':cuSpecHeat[dd]}])
        var_array.append([cuDensity[dd],cuThermCond[dd],cuSpecHeat[dd]])
        #for tt in cuThermCond:
        #    for ss in cuSpecHeat:
        #        # Needs to be list[list[dict]] - outer list is simulation iteration,
        #        # inner list is what is passed to each runner/inputmodifier
        #        moose_vars.append([{'cuDensity':dd,'cuThermCond':tt,'cuSpecHeat':ss}])
        #        var_array.append([dd,tt,ss])
    var_array = np.array(var_array)
    
    print(f"Variable array shape = {var_array.shape}")
    columns = moose_vars[0][0].keys()
    var_df = pd.DataFrame(var_array,columns=columns)
    if save_vals:
        var_path = OUTPUT_DIR / 'moose_vars.csv' #+ Path("moose_vars.csv")
        var_df.to_csv(var_path,columns=columns,index=False)
    
    print('Herd sweep variables:')
    for vv in moose_vars:
        print(vv)
        
    if run_moose:
        
        print("-"*80)
        print('EXAMPLE: Run MOOSE in sequence')
        #print('EXAMPLE: Run MOOSE in parallel')
        print("-"*80)
        
        herd.run_sequential(moose_vars)

        print(f'Run time (seq) = {herd.get_sweep_time():.3f} seconds')
        print("-"*80)
        print()
    
    ### Insert loop here
    if read_results:
        # Get clean results
        for nn in range(len(moose_vars)):
    
            # Use mooseherder to read the exodus and get a SimData object
            data_path = Path(f'examples/sim-workdir-1/sim-1-{nn+1}_out.e') #Path('data/examplesims/plate_2d_thermal_out.e')
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

            if nn==0:
                # locations on our simulation geometry.
                pv_sens = tc_array.get_visualiser()
                pv_sim = t_field.get_visualiser()
                pv_plot = pyvale.plot_sensors(pv_sim,pv_sens,field_name)
                # We label the temperature scale bar ourselves and can
                pv_plot.add_scalar_bar('Temperature, T [degC]')
                pv_plot.show()

            # Now save the desired data into a csv
            
            if save_vals:
                read_config = data_reader.get_read_config()
                sim_data = data_reader.read_sim_data(read_config)
        
                # save sensor positions
                columns = [f"s{i+1}" for i in range(sens_pos.shape[0])]
                sens_pos_df = pd.DataFrame(np.transpose(sens_pos),columns=columns)
                #print(sens_pos_df)
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
        
        # Get noisy results      
        for nn in range(len(moose_vars)):
    
            # Use mooseherder to read the exodus and get a SimData object
            data_path = Path(f'examples/sim-workdir-1/sim-1-{nn+1}_out.e') #Path('data/examplesims/plate_2d_thermal_out.e')
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
            # Gives a n_sensx3 array of sensor positions where each row is a sensor with
            # coords (x,y,z) - can also just manually create this array
            sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
    
            # Set up uncertainty on sensor position:
            pos_noise = np.random.normal(loc=0.0, scale=0.5e-03, size=sens_pos.shape) # Gaussian dist +/- 5% * sim dimensions (ylims)
            sens_pos[:,:2] = sens_pos[:,:2]+pos_noise[:,:2]

            # Now we create a thermocouple array with with the sensor positions and the
            # temperature field from the simulation
            tc_array = pyvale.ThermocoupleArray(sens_pos,t_field)

            # Setup the UQ functions for the sensors. Here we use the basic defaults
            # which is a uniform distribution for the systematic error which is sampled
            # once and remains constant throughout the simulation time creating an
        #     offset. The max temp in the simulation is ~200degC so this range [lo,hi]
            # should be visible on the time traces.
            tc_array.set_uniform_systematic_err_func(low=-5.0,high=5.0) #(low=-10.0,high=10.0)
            # The default for the random error is a normal distribution here we specify
            # a standard deviation which should be visible on the time traces. Note that
            # the random error is sampled repeatedly for each time step.
            tc_array.set_normal_random_err_func(std_dev=5.0)

            measurements = tc_array.get_measurements()

            if nn==0:
                # locations on our simulation geometry.
                pv_sens = tc_array.get_visualiser()
                pv_sim = t_field.get_visualiser()
                pv_plot = pyvale.plot_sensors(pv_sim,pv_sens,field_name)
                # We label the temperature scale bar ourselves and can
                pv_plot.add_scalar_bar('Temperature, T [degC]')
                pv_plot.show()

            # Now save the desired data into a csv
            
            if save_vals:
                read_config = data_reader.get_read_config()
                sim_data = data_reader.read_sim_data(read_config)
        
                # save sensor positions
                columns = [f"s{i+1}" for i in range(sens_pos.shape[0])]
                sens_pos_df = pd.DataFrame(np.transpose(sens_pos),columns=columns)
                #print(sens_pos_df)
                pos_path = OUTPUT_DIR / f'sens_pos_{nn+1:03}.csv'
                sens_pos_df.to_csv(pos_path,columns=columns,index=False)
            
                # save sensor data
                sens_data = np.concatenate([np.reshape(sim_data.time,(1,len(sim_data.time))),measurements])
                print(sens_data.shape)
                columns = ["time"]
                for i in range(measurements.shape[0]):
                    columns.append(f"s{i+1}")
                sens_data_df = pd.DataFrame(np.transpose(sens_data),columns=columns)
                data_path = OUTPUT_DIR / f'noisy_data_{nn+1:03}.csv'
                sens_data_df.to_csv(data_path,columns=columns,index=False)


if __name__ == '__main__':
    main()
