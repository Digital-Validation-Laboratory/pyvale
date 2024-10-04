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
import seaborn as sns

USER_DIR = Path.home() / 'git/herding-moose/'
PLOT_RES = 1#0#


def main() -> None:
    
    # Set super-directory for examples
    OUTPUT_SUPDIR = USER_DIR / 'pyvale/examples/'

    # Read different examples
    OUTPUT_DIR = OUTPUT_SUPDIR / 'case01_no_err/'
    perf_sp, perf_data = read_files(OUTPUT_DIR)
    
    OUTPUT_DIR = OUTPUT_SUPDIR / 'case01_sim_err_only/'
    sim_err_sp, sim_err_data = read_files(OUTPUT_DIR)
    
    OUTPUT_DIR = OUTPUT_SUPDIR / 'case01_rand_err_only/'
    rand_err_sp, rand_err_data = read_files(OUTPUT_DIR)
    
    OUTPUT_DIR = OUTPUT_SUPDIR / 'case01_sys_err_only/'
    sys_err_sp, sys_err_data = read_files(OUTPUT_DIR)
    
    OUTPUT_DIR = OUTPUT_SUPDIR / 'case01_pos_err_only/'
    pos_err_sp, pos_err_data = read_files(OUTPUT_DIR)
    
    print(perf_data.shape,np.array(range(len(perf_data))).shape)
    
    if PLOT_RES:
        fig,axs = plt.subplots(1,1)
        axs.boxplot(perf_data,sym="",
                  positions=np.array(range(perf_data.shape[1]))*2.0-0.5,
                  boxprops=dict(color="green"))
        axs.boxplot(sim_err_data,sym="",
                  positions=np.array(range(sim_err_data.shape[1]))*2.0+0.5,
                  boxprops=dict(color="red"))
        #axs.set_xticklabels(["Perfect data","Material parameter error"])
        axs.set_xticklabels([])
        axs.set_xticks([])
        axs.set_xlabel("Sensor #")
        axs.set_ylabel(r"Temperature [$\degree$C]")
        plt.show()
        
        
    #avm(sim_data[:,-1],noise_data[:,-1][::100])
    mavm(sim_err_data[:,-1],rand_err_data[:,-1][::100])
    #avu(sim_data[:,-1],noise_data[:,-1][::100])
    
    
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
    


def mavm(model_data,exp_data):
    """
    Calculates the Modified Area Validation Metric.
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
    Sn_ = exp_cdf.quantiles
    
    
    df = len(Sn_)-1
    t_alph = stats.t.ppf(0.95,df)#stats.t.interval(0.95,df)[1]#stats.t(0.025)#.pdf(0)
    print(t_alph)
    Sn_conf = [Sn_ - t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_))),
               Sn_ + t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_)))]
    
    
    #print(Sn_conf)
    
    Sn_Y = exp_cdf.probabilities
    F_Y = model_cdf.probabilities
    
    
    if PLOT_RES:
        # plot empirical cdf with conf. int. cdfs
        fig,axs=plt.subplots(1,1)
        model_cdf.plot(axs,label="model")
        #exp_cdf.plot(axs,label="sensor sim")
        axs.ecdf(exp_cdf.quantiles,label="sensor sim")
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
        
        plot_nice = []
        
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
                while jj*P_F > ii*P_Sn:
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
                        
                    plot_nice.append([Sn[jj],F_[ii],P_F*(ii+1)])
                    
                    ii += 1
                    
                d_rem = (Sn[jj]-F_[ii])*(P_Sn*(jj+1) - P_F*ii)
                if d_rem > 0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem
                
        plot_nice = np.array(plot_nice)
        
        if PLOT_RES:
            
            fig,axs = plt.subplots(1,1)
            model_cdf.plot(axs,label="model")
            exp_cdf.plot(axs,label="sensor sim")
            axs.ecdf(Sn_conf[0],color="k",ls="dashed",label="95% C.I.")
            axs.ecdf(Sn_conf[1],color="k",ls="dashed")
            axs.fill_betweenx(plot_nice[:,2],plot_nice[:,0],plot_nice[:,1],
                              where=plot_nice[:,0]<=plot_nice[:,1],color="b",alpha=0.2,
                              label="Upper C.I., area below")
            axs.fill_betweenx(plot_nice[:,2],plot_nice[:,0],plot_nice[:,1],
                              where=plot_nice[:,0]>plot_nice[:,1],color="r",alpha=0.2, 
                              label="Upper C.I., area above")
            axs.legend()
            axs.set_xlabel(r"Temperature [$\degree$C]")
            axs.set_ylabel("Probability")
            plt.show()
                
        d_conf_plus.append(np.abs(d_plus))
        d_conf_minus.append(np.abs(d_minus))
        
    d_plus = np.nanmax(d_conf_plus)
    d_minus = np.nanmax(d_conf_minus)
    
    
    if PLOT_RES:
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



    
def avu(model_data,exp_data):
    """
    INCOMPLETE
    
    Calculates the AMSE V&V 20 Standard Validation Uncertainty.
    Adapted from Whiting et al., 2023, "Assessment of Model Validation, Calibration, and Prediction Approaches in the Presence of Uncertainty", Journal of Verification, Validation and Uncertainty Quantification, Vol. 8.
    Downloaded from http://asmedigitalcollection.asme.org/verification/article-pdf/8/1/011001/6974199/vvuq_008_01_011001.pdf on 24 May 2024.
    """
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
    
    if PLOT_RES:
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
