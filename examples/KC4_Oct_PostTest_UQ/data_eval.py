'''
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from pprint import pprint
from typing import Any
from pathlib import Path
from sklearn import metrics
from scipy import integrate, stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import *
import os, math
import numpy as np
import pandas as pd

np.random.seed(1000)

USER_DIR = Path.home() / 'git/herding-moose/'
PLOT_RES = 1#

class SimDataset:

    def __init__(self, output_dir, file_name, num_tcs):
        
        print("Reading dataset...",end=" ")
        
        out_file_path = output_dir / file_name
        out_data = pd.read_csv(out_file_path)
        
        self.coil_voltage = np.array(out_data["Coil voltage"])
        self.coil_current = np.array(out_data["Coil Current"])
        self.sample_max_temp = np.array(out_data["Sample Maximum Temperature (degC)"])
        self.coil_max_temp = np.array(out_data["Coil Maximum Temperature (degC)"])
        tc_all = []
        for i in range(num_tcs):
            tc_all.append(out_data[f"TC{i+1} (degC)"])
        self.tc_all = np.array(tc_all)
        
        print("Done.")
        
        return None
        
class UQEvalPtsDataset:
    
    def __init__(self, output_dir, file_name, tc_nums, driver):
        
        print("Reading dataset...",end=" ")
        
        out_file_path = output_dir / file_name
        out_data = pd.read_csv(out_file_path)
        
        if driver == "current":
            self.coil_current = np.array(out_data["Coil current"])
        elif driver == "voltage":
            self.coil_voltage = np.array(out_data["Coil voltage"])
        tc_all = []
        for i in tc_nums:
            tc_all.append(out_data[f"Temperature (degC), TC{i}"])
        self.tc_all = np.array(tc_all)
        self.tc_inds = np.array(tc_nums)
        
        print("Done.")
        
        return None

class SteadyDataset:
    
    def __init__(self, output_dir, hive_file_name, dic_file_name, tc_nums):
        
        print("Reading dataset...",end=" ")
        
        hive_file_path = output_dir / hive_file_name
        hive_data = pd.read_csv(hive_file_path)
        
        dic_file_path = output_dir / dic_file_name
        dic_data = pd.read_csv(dic_file_path)
        
        self.coil_current = np.array(hive_data["Coil RMS Current"])
        self.coil_voltage = np.array(hive_data["Coil RMS Voltage"])
        tc_all = []
        tc_source = []
        for i in tc_nums:
            if f"TC{i}" in hive_data.keys().to_list():
                tc_all.append(np.array(hive_data[f"TC{i}"]))
                tc_source.append("H")
            elif f"TC{i}" in dic_data.keys().to_list():
                tc_all.append(np.array(dic_data[f"TC{i}"]))
                tc_source.append("D")
        self.tc_all = tc_all#np.array(tc_all)
        self.tc_inds = np.array(tc_nums)
        self.tc_source = np.array(tc_source)
        
        self.dic_time = np.array(dic_data["Timestamp"])
        self.hive_time = np.array(hive_data["Timestamp"])
        
        fig,axs = plt.subplots(3)#,sharex=True)
        fig.suptitle("Steady state experimental data")
        #axs[0].plot(self.hive_time,self.coil_current)
        #axs[0].set_ylabel("Coil current [A]")
        #axs[1].plot(self.hive_time,self.coil_voltage)
        #axs[1].set_ylabel("Coil voltage [V]")
        axs[0].hist(self.coil_current,histtype="step")
        axs[0].set_xlabel("Coil current [A]")
        axs[1].hist(self.coil_voltage, histtype="step")
        axs[1].set_xlabel("Coil voltage [V]")
        for ii in range(len(tc_nums)):
            if tc_source[ii]=="H":
                #axs[2].plot(self.hive_time,self.tc_all[ii],label=f"TC{tc_nums[ii]}")
                axs[2].hist(self.tc_all[ii],label=f"TC{tc_nums[ii]}",histtype="step")
            elif tc_source[ii]=="D":
                #axs[2].plot(self.dic_time,self.tc_all[ii],label=f"TC{tc_nums[ii]}")
                axs[2].hist(self.tc_all[ii],label=f"TC{tc_nums[ii]}",histtype="step")
        axs[2].legend(ncol=2)
        #axs[2].set_xlabel("Time [s]")
        #axs[2].set_ylabel("Temperature [$\degree$C]")
        axs[2].set_xlabel("Temperature [$\degree$C]")
        fig.tight_layout()
        plt.show()
        
        print("Done")
        
        return None
        
        
        
class ExpCombDataset:
    
    def __init__(self, output_dir, file_name, num_tcs):
        
        print("Reading dataset...",end=" ")
        
        out_file_path = output_dir / file_name
        out_data = pd.read_csv(out_file_path)
        
        print(out_data.keys().to_list())
        
        self.time = np.array(out_data["TimeStamp"])
        tc_all = []
        tc_inds = []
        for i in range(num_tcs):
            if i==0:
                tc_inds.append(i+1)
                tc_all.append(out_data[f" K-Type TC [degC]"])
            elif f" K-Type TC {i+1} [degC]" in out_data.keys().to_list() and (~np.any(out_data[f" K-Type TC {i+1} [degC]"]=="Bad Thermocouple") and ~np.all(out_data[f" K-Type TC {i+1} [degC]"]==0)):
                tc_inds.append(i+1)
                tc_all.append(out_data[f" K-Type TC {i+1} [degC]"])
        self.tc_all = np.array(tc_all)
        self.tc_inds = np.array(tc_inds)
        
        print("Done.")
        
        return None

def main():
    
    OUTPUT_SUPDIR = USER_DIR / 'pyvale/examples/'
    OUTPUT_DIR = OUTPUT_SUPDIR / "KC4_Oct_PostTest_UQ/"
    SIM_FILE_NAME = "KC4_Oct_PostTest_UQ.csv"
    
    # Focus on test 38
    # Trusted TCs given by JHJ: 1,2,3,4,5,8
    # GIVES DIFF VALUES IN STEADY SUMMARY THO...
    tc_nums = [1,2,4,5,7,8]#[1,2,3,4,5,8]
    
    # Read data from model (forward UQ) and experiments
    CURR_FILE_NAME = "KC4 Phase 1 Test 38 Current UQ Evaluated Points.csv"
    currRunUQ = read_exp_data(OUTPUT_DIR,CURR_FILE_NAME,tc_nums,driver="current")
    
    VOLT_FILE_NAME = "KC4 Phase 1 Test 38 Voltage UQ Evaluated Points.csv"
    voltRunUQ = read_exp_data(OUTPUT_DIR,VOLT_FILE_NAME,tc_nums,driver="voltage")
    
    HIVE_FILE_NAME = "38_HIVE_SteadyData.csv"
    DIC_FILE_NAME = "38_DIC_SteadyData.csv"
    steadyData = SteadyDataset(OUTPUT_DIR, HIVE_FILE_NAME, DIC_FILE_NAME, tc_nums)
    
    print("Mean values\nTC ID\tModel\tExperiment")
    for tt in range(len(tc_nums)):
        print(f"TC{tc_nums[tt]}\t{np.nanmean(currRunUQ.tc_all[tt]):.2e}\t{np.nanmean(steadyData.tc_all[tt]):.2e}")
    print("Std dev\nTC ID\tModel\tExperiment")
    for tt in range(len(tc_nums)):
        print(f"TC{tc_nums[tt]}\t{np.nanstd(currRunUQ.tc_all[tt]):.2e}\t{np.nanstd(steadyData.tc_all[tt]):.2e}")
    
    x = 1000*np.array(range(len(tc_nums)))
    
    plt.figure()
    plt.boxplot([tc.flatten() for tc in currRunUQ.tc_all],labels=[f"TC{tc_num}" for tc_num in tc_nums],sym="",positions=x-100,widths=150)
    plt.boxplot([tc.flatten() for tc in steadyData.tc_all],labels=None,sym="",positions=x+100,widths=150)
    plt.xticks(x,labels=[f"TC{tc_num}" for tc_num in tc_nums])
    plt.ylabel("Temperature [$\degree$C]")
    plt.show()
    
    # Validation metric for current-driven UQ
    plt.figure()
    plt.boxplot([currRunUQ.coil_current,steadyData.coil_current],labels=["model","experiment"])
    plt.ylabel("Coil current [A]")
    plt.show()
    for tt in range(len(tc_nums)):
        print(f"TC{tc_nums[tt]}")
        mavm_output = mavm(currRunUQ.tc_all[tt],steadyData.tc_all[tt])
        print(mavm_output["d+"],mavm_output["d-"])
    
    # Validation metric for voltage-driven UQ
    plt.figure()
    plt.boxplot([voltRunUQ.coil_voltage,steadyData.coil_voltage],labels=["model","experiment"])
    plt.ylabel("Coil voltage [V]")
    plt.show()
    for tt in range(len(tc_nums)):
        print(f"TC{tc_nums[tt]}")
        mavm_output = mavm(voltRunUQ.tc_all[tt],steadyData.tc_all[tt])
        print(mavm_output["d+"],mavm_output["d-"])
    
    return None
    
    
    
    
    
def read_sim_data(output_dir,file_name,num_tcs,driver):
    
    dataset = SimDataset(output_dir,file_name,num_tcs)
    
    print("TC dataset size:",dataset.tc_all.shape)
    
    if driver == "voltage":
        plt.figure()
        for ii in range(dataset.tc_all.shape[0]):
            plt.plot(dataset.coil_voltage,dataset.tc_all[ii],".",label=f"TC{ii+1}")
        plt.plot(dataset.coil_voltage,dataset.sample_max_temp,"k.",label=f"Sample max temp")
        plt.legend()
        plt.xlabel("Coil voltage [V]")
        plt.ylabel("Thermocouple reading [$\degree$C]")
        plt.show()
        
        #rounded_voltage = np.rint(dataset.coil_voltage)
        voltage_values = np.unique(dataset.coil_voltage)
        #voltage_values = np.unique(rounded_voltage)
        
        tc_temps = []
        max_temps = []
        for vv in voltage_values:
            voltage_indices = np.argwhere(dataset.coil_voltage==vv).flatten()
            #voltage_indices = np.argwhere(rounded_voltage==vv).flatten()
            tc_temps.append(dataset.tc_all[:,voltage_indices])
            max_temps.append(dataset.sample_max_temp[voltage_indices])
        
        tc_temps = np.array(tc_temps) # tc_temps: [voltage index, thermocouple #, sample #]
        max_temps = np.array(max_temps) # max_temps: [voltage index, sample #]
        
    elif driver == "current":    
        plt.figure()
        for ii in range(dataset.tc_all.shape[0]):
            plt.plot(dataset.coil_current,dataset.tc_all[ii],".",label=f"TC{ii+1}")
        plt.plot(dataset.coil_current,dataset.sample_max_temp,"k.",label=f"Sample max temp")
        plt.legend()
        plt.xlabel("Coil current [A]")
        plt.ylabel("Thermocouple reading [$\degree$C]")
        plt.show()
        
        current_values = np.unique(dataset.coil_current)
        
        tc_temps = []
        max_temps = []
        for vv in current_values:
            current_indices = np.argwhere(dataset.coil_current==vv).flatten()
            tc_temps.append(dataset.tc_all[:,current_indices])
            max_temps.append(dataset.sample_max_temp[current_indices])
        
        tc_temps = np.array(tc_temps) # tc_temps: [voltage index, thermocouple #, sample #]
        max_temps = np.array(max_temps) # max_temps: [voltage index, sample #]
    
    plt.figure()
    for tc_ in range(dataset.tc_all.shape[0]):
        plt.ecdf(tc_temps[:,tc_].flatten(),label=f"TC{tc_+1}")
    plt.ecdf(max_temps.flatten(),label="Max temp")
    plt.legend()
    plt.xlabel(f"Thermocouple reading [$\degree$C]")
    plt.ylabel("Cumulative distribution function")
    plt.show()
    
    if driver=="voltage":
        return voltage_values,tc_temps,max_temps
    elif driver=="current":
        return current_values,tc_temps,max_temps
    
    
def read_exp_data(output_dir,file_name,tc_nums,driver):
    
    dataset = UQEvalPtsDataset(output_dir,file_name,tc_nums,driver)
    
    print("TC dataset size:",dataset.tc_all.shape)
    
    fig,axs = plt.subplots(2)
    if driver=="current":
        fig.suptitle("Current-driven forward UQ")
        axs[0].hist(dataset.coil_current,bins=100,histtype="step")
        axs[0].set_xlabel("Coil current [A]")
    elif driver=="voltage":
        fig.suptitle("Voltage-driven forward UQ")
        axs[0].hist(dataset.coil_voltage,bins=100,histtype="step")
        axs[0].set_xlabel("Coil voltage [V]")
    for ii in range(dataset.tc_all.shape[0]):
        axs[1].hist(dataset.tc_all[ii],bins=100,histtype="step",label=f"TC{dataset.tc_inds[ii]}")
    axs[1].set_xlabel("Temperature [$\degree$C]")
    axs[1].legend()
    fig.tight_layout()
    fig.show()
    
    
    return dataset
    
def read_exp_comb_data(output_dir,file_name,num_tcs,driver):
    
    dataset = ExpCombDataset(output_dir,file_name,num_tcs)
    
    print("TC dataset size:",dataset.tc_all.shape)
    
    plt.figure()
    for ii in range(dataset.tc_all.shape[0]):
        plt.plot(dataset.time,dataset.tc_all[ii],".",label=f"TC{ii+1}")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Thermocouple reading [$\degree$C]")
    plt.show()
    
    return 0
    
    
    
def mavm(model_data,exp_data,plotRes=None):
    """
    Calculates the Modified Area Validation Metric.
    Adapted from Whiting et al., 2023, "Assessment of Model Validation, Calibration, and Prediction Approaches in the Presence of Uncertainty", Journal of Verification, Validation and Uncertainty Quantification, Vol. 8.
    Downloaded from http://asmedigitalcollection.asme.org/verification/article-pdf/8/1/011001/6974199/vvuq_008_01_011001.pdf on 24 May 2024.
    """
    
    if plotRes == None:
        plotRes = PLOT_RES
    
    # find empirical cdf
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf
    
    if plotRes:
        # plot empirical cdf
        fig,axs=plt.subplots(1,1)
        model_cdf.plot(axs,label="model")
        exp_cdf.plot(axs,label="experiment")
        axs.legend()
        axs.set_xlabel(r"Temperature [$\degree$C]")
        axs.set_ylabel("Probability")
        plt.show()
    
    F_ = model_cdf.quantiles
    Sn_ = exp_cdf.quantiles
    
    
    df = len(Sn_)-1
    t_alph = stats.t.ppf(0.95,df)
    
    Sn_conf = [Sn_ - t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_))),
               Sn_ + t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_)))]
    
    
    Sn_Y = exp_cdf.probabilities
    F_Y = model_cdf.probabilities
    
    
    if plotRes:
        # plot empirical cdf with conf. int. cdfs
        fig,axs=plt.subplots(1,1)
        axs.ecdf(model_cdf.quantiles,label="model")
        axs.ecdf(exp_cdf.quantiles,label="experiment")
        axs.ecdf(Sn_conf[0],ls="dashed",color="k",label="95% C.I.")
        axs.ecdf(Sn_conf[1],ls="dashed",color="k")
        axs.legend()
        axs.set_xlabel(r"Temperature [$\degree$C]")
        axs.set_ylabel("Probability")
        plt.show()
    
    
    P_F = 1/len(F_)
    P_Sn = 1/len(exp_cdf.quantiles)
    
    d_conf_plus = []
    d_conf_minus = []
    
    for k in [0,1]:
    
        ii = 0
        d_rem = 0
        
        d_plus = 0
        d_minus = 0
        
        
        Sn = Sn_conf[k]
        
        #If more experimental data points than model data points
        if len(Sn) > len(F_):
            
            for jj in range(0,len(F_)):
                if d_rem != 0:
                    d_ = (Sn[ii] - F_[jj]) * (P_Sn*(ii+1) - P_F*jj)
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_
                    ii += 1
                while (jj+1)*P_F > (ii+1)*P_Sn:
                    d_ = (Sn[ii] - F_[jj])*P_F
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_
                    
                    ii += 1
                d_rem = (Sn[ii]-F_[jj])*(P_F*(jj+1) - P_Sn*ii)
                if d_rem > 0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem
                   
        #If more model data points than experimental data points (more typical) 
        elif len(Sn) <= len(F_):
            
            for jj in range(0,len(Sn)):
            
                if d_rem != 0:
                    d_ = (Sn[jj]-F_[ii])*(P_F*(ii+1) - P_Sn*jj)
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_
                    ii += 1
                    
                while (ii+1)*P_F < (jj+1)*P_Sn:
                    d_ = (Sn[jj]-F_[ii])*P_F
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_
                    
                    ii += 1
                    
                d_rem = (Sn[jj]-F_[ii])*(P_Sn*(jj+1) - P_F*ii)
                if d_rem > 0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem
                
        d_conf_plus.append(np.abs(d_plus))
        d_conf_minus.append(np.abs(d_minus))
        
    d_plus = np.nanmax(d_conf_plus)
    d_minus = np.nanmax(d_conf_minus)
    
    
    if plotRes:
        plt.figure()
        plt.plot(F_,F_Y,"k-")
        plt.plot(F_+d_plus,F_Y,"k--")
        plt.plot(F_-d_minus,F_Y,"k--")
        plt.fill_betweenx(F_Y,F_-d_minus,F_+d_plus,color="k",alpha=0.2)
        plt.xlabel(r"Temperature [$\degree$C]")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.show()
        
    output_dict = {"model_cdf":model_cdf,
                   "exp_cdf":exp_cdf,
                   "d+":d_plus,
                   "d-":d_minus}
    
    return output_dict
    
    
if __name__ == "__main__":
    main()
    
