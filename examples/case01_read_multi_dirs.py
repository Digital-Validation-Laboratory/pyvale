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
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import *
import os, math
import numpy as np
import pandas as pd

np.random.seed(1000)

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
    
    dataset_names = ["no_err","sim_err","rand_err","sys_err","pos_err"]
    
    for dataset_name in dataset_names:
        OUTPUT_DIR = OUTPUT_SUPDIR / f"case01_{dataset_name}/"
        if dataset_name in ("no_err","sim_err"):#
        #if dataset_name in ("no_err"):
            datasets[dataset_name] = Dataset(dataset_name,OUTPUT_DIR)
        else:
            datasets[dataset_name] = Dataset(dataset_name,OUTPUT_DIR,skip=100)
    
    print("Done.")
    
    # Pick datasets to compare
    
    data1 = datasets["sim_err"].data
    data1_label=datasets["sim_err"].name
    
    data2 = datasets["rand_err"].data
    data2_label = datasets["rand_err"].name
    
    # Plot two datasets against each other
    
    if PLOT_RES:
        fig,axs = plt.subplots(1,1)
        bp1 = axs.boxplot(data1,sym="",
                          positions=np.array(range(data1.shape[1]))*2.0-0.5,
                          boxprops=dict(color="green"),medianprops=dict(color="green"))
        bp2 = axs.boxplot(data2,sym="",
                          positions=np.array(range(data2.shape[1]))*2.0+0.5,
                          boxprops=dict(color="red"),medianprops=dict(color="red"))
        axs.legend([bp1["boxes"][0], bp2["boxes"][0]], [data1_label,data2_label])
        axs.set_xticklabels([])
        axs.set_xticks([])
        axs.set_xlabel("Sensor #")
        axs.set_ylabel(r"Temperature [$\degree$C]")
        plt.show()
        
    # Test validation metrics
    
    #avm(sim_data[:,-1],noise_data[:,-1])
    mavm(data1[:,-1],data2[:,-1])
    #reliability_metric(data1[:,-1],data2[:,-1],10)
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
        #plt.show()
    
    F_ = model_cdf.quantiles
    Sn_ = exp_cdf.quantiles
    
    
    df = len(Sn_)-1
    t_alph = stats.t.ppf(0.95,df)
    
    Sn_conf = [Sn_ - t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_))),
               Sn_ + t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_)))]
    
    
    Sn_Y = exp_cdf.probabilities
    F_Y = model_cdf.probabilities
    
    
    if PLOT_RES:
        # plot empirical cdf with conf. int. cdfs
        fig,axs=plt.subplots(1,1)
        axs.ecdf(model_cdf.quantiles,label="model")
        axs.ecdf(exp_cdf.quantiles,label="synthetic experiment")
        axs.ecdf(Sn_conf[0],ls="dashed",color="k",label="95% C.I.")
        axs.ecdf(Sn_conf[1],ls="dashed",color="k")
        axs.legend()
        axs.set_xlabel(r"Temperature [$\degree$C]")
        axs.set_ylabel("Probability")
        #plt.show()
    
    
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
    
    
    if PLOT_RES:
        fig, axs = plt.subplots(1)
        plt.plot(F_,F_Y,"k-")
        plt.plot(F_+d_plus,F_Y,"k--")
        plt.plot(F_-d_minus,F_Y,"k--")
        plt.fill_betweenx(F_Y,F_-d_minus,F_+d_plus,color="k",alpha=0.2)
        axs.ecdf(exp_cdf.quantiles,label="synthetic experiment")
        axs.ecdf(Sn_conf[0],ls="dashed",color="k",label="95% C.I.")
        axs.ecdf(Sn_conf[1],ls="dashed",color="k")
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
    
    
def reliability_metric(model_data,exp_data,eta):
    """
    Returns probability of difference between observed data and model prediction being < a given tolerance limit eta (E)
    r = P(-E < d < E), d = Y_D - Y_m
    Two factors: eta helps to estimate the probability, but the adequacy requirement is c, where we accept the model prediction only when P(-E<D<E)>=c
    Model prediction is a single number, data are replicated experimental measurements taken for the same input.
    """
    # phi = probability density function
    
    # find empirical cdf
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf
    
    Sn_ = exp_cdf.quantiles
    Sn_Y = exp_cdf.probabilities
    F_ = model_cdf.quantiles
    F_Y = model_cdf.probabilities
    
    # rel-based metric (Ling & Mahadevan)
    mean_model = np.nanmean(model_data)
    mean_exp = np.nanmean(exp_data)
    
    #eta_plus = np.sqrt(len(exp_data))*(eta-(mean_exp-mean_model))/np.nanstd(exp_data)
    #eta_minus = np.sqrt(len(exp_data))*(-1*eta - (mean_exp-mean_model))/np.nanstd(exp_data)
    #r = exp_cdf.evaluate(mean_exp+eta_plus) - exp_cdf.evaluate(mean_exp+eta_minus)
    # ?? Must have misunderstood - try different method

    # diff method
    diff_matrix = np.zeros(len(model_data)*len(exp_data))
    
    for N in range(len(exp_data)):
        diff_matrix[N*len(model_data):(N+1)*len(model_data)] = model_data-exp_data[N]
   
    diff_cdf = stats.ecdf(diff_matrix).cdf
    
    if PLOT_RES:
        fig,axs=plt.subplots(1)
        diff_cdf.plot(axs)
        plt.vlines([eta/2,-eta/2],0,1,color="k")
        plt.xlabel("Expectation difference")
        plt.ylabel("Probability")
        plt.show()
    
    r = diff_cdf.evaluate(eta/2) - diff_cdf.evaluate(-eta/2)
    print("r =",r)


    return r
    
def baycal(model_data,exp_data):
    # from: https://ndcbe.github.io/cbe67701-uncertainty-quantification/11.03-Contributed-Example.html
    
    # covariance function
    def cov(x,y,beta,l,alpha):
        exponent = np.sum(beta*np.abs(x-y)**alpha)
        return 1/l*math.exp(-exponent)
    
    #likelihood function
    def likelihood(z,x,beta,lam,alpha, beta_t, lam_t, alpha_t, meas_cov, N,M):
        Sig_z = np.zeros((N+M,N+M))
        #fill in matrix with sim covariance
        for i in range(N+M):
            for j in range(i+1):
                tmp = cov(x[i,:],x[j,:],beta,lam,alpha)
                if (i < N):
                    tmp += cov(x[i,0], x[j,0], beta_t, lam_t, alpha_t)
                Sig_z[i,j] = tmp
                Sig_z[j,i] = tmp
        #add in measurement error cov
        Sig_z[0:N,0:N] += meas_cov
        #print(Sig_z)
        likelihood = stats.multivariate_normal.logpdf(z,mean=0*z, cov=Sig_z,allow_singular=True)
        return likelihood
    
    M = len(model_data)
    N = len(exp_data)
    x = np.zeros((N+M,2))
    
    x[0:N,0] = model_data
    x[N:(N+M),:]  = exp_data
    
    return None

if __name__ == '__main__':
    main()
