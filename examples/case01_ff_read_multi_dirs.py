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
from sklearn import metrics
from scipy import integrate, stats
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

USER_DIR = Path.home() / 'git/herding-moose/'
PLOT_RES = 1#


class Dataset:

    def __init__(self,name,output_dir,skip=None):
        
        print("Reading dataset...",end=" ")
        self.name = name
        self.sens_pos,self.data = self.read_files(output_dir,skip=skip)
        print("Done.")
        
        return None
        
    def read_files(self,output_dir,skip=None):
        file_list = os.listdir(output_dir)

        pos_files = list([])
        out_files = list([])
    
        for ff,file in enumerate(file_list):
            if file[:8] == "sens_pos":
                pos_files.append(file)
            if file[:8] == "sim_data":
                out_files.append(file)

        sens_pos = list([])
        sim_data = list([])
    
        for ff in range(len(pos_files)):
    
            pos_file_path = output_dir / f"sens_pos_{ff+1:03}.csv"
            pos_data = pd.read_csv(pos_file_path)
            sens_pos.append(pos_data)
        
            out_file_path = output_dir / f"sim_data_{ff+1:03}.csv"
            out_data = pd.read_csv(out_file_path)
            sim_data.append(np.array([list(out_data[f"s{yy+1}"])[-1] for yy in range(len(out_data.columns)-1)]))
        
        sens_pos = np.array(sens_pos)
        sim_data = np.array(sim_data)
        
        if skip != None:
            sens_pos = sens_pos[::skip]
            sim_data = sim_data[::skip]
    
        return sens_pos, sim_data


def main():
    
    # Set super-directory for examples
    OUTPUT_SUPDIR = USER_DIR / 'pyvale/examples/'
    datasets = dict()
    
    # run matrix?
    runMatrix = 0#1#

    # Read different examples
    print("Reading example cases...",end=" ")
    
    dataset_names = ["no_err","sim_err","rand_err"]#["no_err","sim_err","rand_err","sys_err","pos_err"]
    
    for dataset_name in dataset_names:
        OUTPUT_DIR = OUTPUT_SUPDIR / f"case01_ff_{dataset_name}/"
        if dataset_name in ("no_err","sim_err"):
            datasets[dataset_name] = Dataset(dataset_name,OUTPUT_DIR)
        else:
            datasets[dataset_name] = Dataset(dataset_name,OUTPUT_DIR,skip=100)
    
    print("Done.")
    
    # Pick datasets to compare
    
    data1 = np.array(datasets["sim_err"].data)
    data1_pos = np.array(datasets["sim_err"].sens_pos)
    data1_label=datasets["sim_err"].name
    
    data2 = np.array(datasets["rand_err"].data)
    data2_pos = np.array(datasets["rand_err"].sens_pos)
    data2_label = datasets["rand_err"].name
    
    # Plot two datasets against each other
    data1_mean = np.nanmean(data1,axis=0)
    data1_std = np.nanstd(data1,axis=0)
    data1_pos_mean = np.nanmean(data1_pos,axis=0)
    
    data2_mean = np.nanmean(data2,axis=0)
    data2_std = np.nanstd(data2,axis=0)
    data2_pos_mean = np.nanmean(data2_pos,axis=0)
    
    if PLOT_RES:
        # plot mean
        fig,axs = plt.subplots(2,1)
        cm1 = axs[0].tricontourf(data1_pos_mean[0],data1_pos_mean[1],
                                  data1_mean.flatten(),cmap="plasma",levels=100)
        fig.colorbar(cm1,ax=axs[0])
        cm2 = axs[1].tricontourf(data2_pos_mean[0],data2_pos_mean[1],
                                  data2_mean.flatten(),cmap="plasma",levels=100)
        fig.colorbar(cm2,ax=axs[1])
        fig.suptitle("Mean Temperature")
        plt.show()
        
        # plot std
        fig,axs = plt.subplots(2,1)
        cm1 = axs[0].tricontourf(data1_pos_mean[0],data1_pos_mean[1],
                                  data1_std.flatten(),cmap="plasma",levels=100)
        fig.colorbar(cm1,ax=axs[0])
        cm2 = axs[1].tricontourf(data2_pos_mean[0],data2_pos_mean[1],
                                  data2_std.flatten(),cmap="plasma",levels=100)
        fig.colorbar(cm2,ax=axs[1])
        fig.suptitle("Std Dev Temperature")
        plt.show()
        
        
    # Test validation metrics
    
    nsamples, npoints = data1.shape
    
    data_d_plus = np.full(npoints,0)
    data_d_minus = np.full(npoints,0)
    
    for nn in range(npoints):
        mavm_out = mavm(data1[:,nn],data2[:,nn],plotRes=0)
        data_d_plus[nn] = mavm_out["d+"]
        data_d_minus[nn] = mavm_out["d-"]
        if nn ==0:
            print(data1[:,nn].shape)
            print(mavm_out["d+"])
        
    print(data_d_plus)
    print(data_d_minus)
        
    fig,axs = plt.subplots(2,1,sharex=True,sharey=True)
    cm1 = axs[0].tricontourf(data1_pos_mean[0],data1_pos_mean[1],
                             data_d_plus.flatten(),cmap="plasma",levels=100)
    axs[0].set_title("d+")
    cm2 = axs[1].tricontourf(data1_pos_mean[0],data1_pos_mean[1],
                             data_d_minus.flatten(),cmap="plasma",levels=100)
    axs[1].set_title("d-")
    fig.colorbar(cm2,ax=axs.tolist(),label="Temperature uncertainty")
    #fig.suptitle("MAVM d+/d-")
    plt.show()
    
    #avm(sim_data[:,-1],noise_data[:,-1])
    #mavm(data1[:,-1],data2[:,-1])
    #avu(sim_data[:,-1],noise_data[:,-1])
    
    # Create matrix
    # Note: columns are considered to be model data; rows as synthetic experiments
    
    if runMatrix:
        # Get d+ table
        print("d+")
        comp_table = {"Datasets":[ds_name for ds_name in datasets.keys()]}
        
        for ds1_name in ("no_err","sim_err"):#datasets.keys():
            comp_list = []
            for ds2_name in datasets.keys():
                ds1 = datasets[ds1_name]
                ds2 = datasets[ds2_name]
                comp_list.append(mavm(ds1.data[:,-1],ds2.data[:,-1])["d+"])
            comp_table[ds1_name] = comp_list
        comp_panda = pd.DataFrame(data=comp_table)
        print(comp_panda)
        
        # Get d- table
        print("d-")
        comp_table = {"Datasets":[ds_name for ds_name in datasets.keys()]}
        
        for ds1_name in ("no_err","sim_err"):#datasets.keys():
            comp_list = []
            for ds2_name in datasets.keys():
                ds1 = datasets[ds1_name]
                ds2 = datasets[ds2_name]
                comp_list.append(mavm(ds1.data[:,-1],ds2.data[:,-1])["d-"])
            comp_table[ds1_name] = comp_list
        comp_panda = pd.DataFrame(data=comp_table)
        print(comp_panda)
    
    
def read_files(file_directory):
    
    file_list = os.listdir(file_directory)

    pos_files = list([])
    out_files = list([])
    
    for ff,file in enumerate(file_list):
        if file[:8] == "sens_pos":
            pos_files.append(file)
        if file[:8] == "sim_data":
            out_files.append(file)

    sens_pos = list([])
    sim_data = list([])
    
    for ff in range(len(pos_files)):
    
        pos_file_path = file_directory / f"sens_pos_{ff+1:03}.csv"
        pos_data = pd.read_csv(pos_file_path)
        sens_pos.append(pos_data)
        
        out_file_path = file_directory / f"sim_data_{ff+1:03}.csv"
        out_data = pd.read_csv(out_file_path)
        sim_data.append(np.array([list(out_data[f"s{yy+1}"])[-1] for yy in range(len(out_data.columns)-1)]))
        
    sens_pos = np.array(sens_pos)
    sim_data = np.array(sim_data)
    
    return sens_pos, sim_data

    
def avm(model_data,exp_data):
    """
    Calculates the Area Validation Metric.
    Adapted from Whiting et al., 2023, "Assessment of Model Validation, Calibration, and Prediction Approaches in the Presence of Uncertainty", Journal of Verification, Validation and Uncertainty Quantification, Vol. 8.
    Downloaded from http://asmedigitalcollection.asme.org/verification/article-pdf/8/1/011001/6974199/vvuq_008_01_011001.pdf on 24 May 2024.
    """
    
    # find empirical cdf
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf
    
    if PLOT_RES:
        # plot empirical cdf
        fig,axs=plt.subplots(1,1)
        model_cdf.plot(axs,label="model")
        exp_cdf.plot(axs,label="sensor sim")
        axs.legend()
        axs.set_xlabel(r"Temperature [$\degree$C]")
        axs.set_ylabel("Probability")
        plt.show()
        
    F_ = model_cdf.quantiles
    Sn = exp_cdf.quantiles
    
    Sn_Y = exp_cdf.probabilities
    F_Y = model_cdf.probabilities
    
    P_F = 1/len(F_)
    P_Sn = 1/len(Sn)
    
    d_diff = 0
    
    ii = 0
    d_rem = 0
    
    if len(Sn) > len(F_):
        #If more experimental data points than model data points
        for jj in range(0,len(F_)):
            if d_rem != 0:
                d_ = (Sn[ii] - F_[jj]) * (P_Sn*(ii+1) - P_F*jj)
                
                d_diff += np.abs(d_)
                ii += 1
                
            while jj*P_F > ii*P_Sn:
                d_ = (Sn[ii] - F_[jj])*P_F
                
                d_diff += np.abs(d_)
                ii += 1
                
            d_rem = (Sn[ii]-F_[jj])*(P_F*(jj+1) - P_Sn*ii)
            
            d_diff += np.abs(d_rem)
                
    elif len(Sn) <= len(F_):
        #If more model data points than experimental data points (more typical)
        for jj in range(0,len(Sn)):
            if d_rem != 0:
                d_ = (Sn[jj]-F_[ii])*(P_F*(ii+1) - P_Sn*jj)
                
                d_diff += np.abs(d_)
                ii += 1
            while ii*P_F < jj*P_Sn:
                d_ = (Sn[jj]-F_[ii])*P_F
                
                d_diff += np.abs(d_)
                ii += 1
            d_rem = (Sn[jj]-F_[ii])*(P_Sn*(jj+1) - P_F*ii)
            
            d_diff += np.abs(d_rem)
                
    d_diff = np.abs(d_diff)
    
    if PLOT_RES:
        plt.figure()
        plt.plot(F_,F_Y,"k-")
        plt.plot(F_+d_diff,F_Y,"k--")
        plt.plot(F_-d_diff,F_Y,"k--")
        plt.fill_betweenx(F_Y,F_-d_diff,F_+d_diff,color="k",alpha=0.2)
        plt.xlabel(r"Temperature [$\degree$C]")
        plt.ylabel("Probability")
        plt.show()
    
    output_dict = {"model_cdf":model_cdf,
                   "exp_cdf":exp_cdf,
                   "d+":d_diff}
    
    return output_dict
    


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
        exp_cdf.plot(axs,label="sensor sim")
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
        axs.ecdf(exp_cdf.quantiles,label="synthetic experiment")
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
        plt.show()
        
    output_dict = {"model_cdf":model_cdf,
                   "exp_cdf":exp_cdf,
                   "d+":d_plus,
                   "d-":d_minus}
    
    return output_dict



    
def avu(model_data,exp_data,plotRes=None):
    """
    INCOMPLETE
    
    Calculates the AMSE V&V 20 Standard Validation Uncertainty.
    Adapted from Whiting et al., 2023, "Assessment of Model Validation, Calibration, and Prediction Approaches in the Presence of Uncertainty", Journal of Verification, Validation and Uncertainty Quantification, Vol. 8.
    Downloaded from http://asmedigitalcollection.asme.org/verification/article-pdf/8/1/011001/6974199/vvuq_008_01_011001.pdf on 24 May 2024.
    """
    
    if plotRes == None:
        plotRes = PLOT_RES
    
    # find empirical cdf
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf
    
    Sn_ = exp_cdf.quantiles
    F_ = model_cdf.quantiles
    F_Y = model_cdf.probabilities
    
    E_ = np.nanmean(Sn_) - np.nanmean(F_)
    
    df = len(Sn_)-1
    t_alph = stats.t.ppf(0.95,df)
    print(t_alph)
    
    u_D = t_alph * np.nanstd(Sn_)/np.sqrt(len(Sn_))
    u_input = np.nanstd(F_)
    u_num = 0 #PLACEHOLDER #u_ro + u_iter + u_DE
    print(u_D,u_input,u_num)
    
    u_val = np.sqrt(u_D**2 + u_input**2 + u_num**2)
    
    if plotRes:
        plt.figure()
        plt.plot(F_,F_Y,"k-")
        plt.plot(F_+u_val,F_Y,"k--")
        plt.plot(F_-u_val,F_Y,"k--")
        plt.fill_betweenx(F_Y,F_-u_val,F_+u_val,color="k",alpha=0.2)
        plt.xlabel(r"Temperature [$\degree$C]")
        plt.ylabel("Probability")
        plt.show()
    
    output_dict = {"model_cdf":model_cdf,
                   "exp_cdf":exp_cdf,
                   "u":u_val}

    return output_dict
    
def baycal(model_data,exp_data):
    
    # See Whiting et al; Kennedy & O'Hagan.
    
    return None

if __name__ == '__main__':
    main()
