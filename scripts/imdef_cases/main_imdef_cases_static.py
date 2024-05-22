"""
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================

GENERATE IMAGE DEFORMATION TEST CASES
"""

import os
import time
from pprint import pprint
import pickle
from pathlib import Path

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import pyvale.imagesim.asysreader as ar

# DEFINITIONS
diag_on = True
[xi,yi] = [0,1]
# Colour map for images and vector fields
im_cmap = 'gray'
v_cmap = 'plasma'

class TestCase:
    def __init__(self):
        self.test_num = 1
        self.test_tag = ''
        self.test_val = 0.0
        self.geom = [78e-3,48e-3]
        self.elem_size = [0.25e-3,0.25e-3]
        self.m_per_px = 1.0e-3/5

def create_test_case(save_path,test_case,fe_data,gen_data):
    print('')
    print('--------------------------------------------------------------------')
    print('TEST CASE {}'.format(test_case.test_num))
    print('Description: {}'.format(test_case.test_tag))
    print('')
    print('Creating test case.')

    # Generic test params
    fe_data = gen_data
    fe_data.params.test_num = test_case.test_num
    fe_data.params.descript = test_case.test_tag

    # Make sure the displacement appears to be 2D with disp as columns
    if fe_data.disp.x.ndim == 1:
        fe_data.disp.x = np.atleast_2d(fe_data.disp.x).T
        fe_data.disp.y = np.atleast_2d(fe_data.disp.y).T

    # Create new directory to save the test case to
    test_path = save_path / str('imdefcase{}_'.format(test_case.test_num)+test_case.test_tag)

    print('Saving test case data to:')
    print('{}'.format(test_path))
    if not test_path.is_dir():
        test_path.mkdir()

    # Pickle the synthetic fe_data class
    tic = time.time()
    with open(test_path / 'fe_data.pkl','wb') as fe_save_file:
        pickle.dump(fe_data,fe_save_file)
    toc = time.time()
    print('Saving data files took {:.4f} seconds'.format(toc-tic))


def main() -> None:
    # LOAD DATA
    print('')
    print('--------------------------------------------------------------------')
    print('GENERATE IMAGE DEFORMATION TEST CASES')
    print('--------------------------------------------------------------------')
    # Gets the directory of the current script file
    cwd = Path.cwd()
    print("Current working directory:")
    print(cwd)

    save_path = Path.cwd() / 'scripts' / 'imdef_cases'
    print("Save directory:")
    print(save_path)

    test_case = TestCase()
    pprint(vars(test_case))

    #
    print('')
    print('--------------------------------------------------------------------')
    print('CREATING GENERIC TEST CASE')
    print('')
    print('Creating nodal array and zero frame data.')

    # Create vectors of nodal values and create a 2D mesh grid
    node_vec_x = np.arange(0,test_case.geom[xi]+test_case.elem_size[xi],
                        test_case.elem_size[xi])
    node_vec_y = np.arange(0,test_case.geom[yi]+test_case.elem_size[yi],
                        test_case.elem_size[yi])
    [node_grid_x,node_grid_y] = np.meshgrid(node_vec_x,node_vec_y)
    node_grid_y = node_grid_y[::-1,:] # flipud

    print('Creating empty FE data object.')
    gen_data = ar.FEData(cwd,0,2)
    #pprint(vars(gen_data))

    gen_data.nodes.loc_x = node_grid_x.flatten()
    gen_data.nodes.loc_y = node_grid_y.flatten()
    gen_data.nodes.loc_z = np.zeros(gen_data.nodes.loc_x.shape)
    gen_data.nodes.nums = np.arange(1,len(gen_data.nodes.loc_x)+1)
    #pprint(vars(gen_data.nodes))

    # TEST CASE 1: Large Rigid Body Motion
    test_case.test_num = 1
    test_case.test_tag = 'RigidBodyMotion_4_1px'
    test_case.test_val = 4.1

    # Create generic data object and unique params
    fe_data = gen_data
    fe_data.params.rigid_disp_px = test_case.test_val
    fe_data.params.rigid_disp_m = test_case.test_val*test_case.m_per_px

    # Init displacement fields
    fe_data.disp.node_nums = fe_data.nodes.nums
    fe_data.disp.x = np.ones(fe_data.nodes.nums.shape)*fe_data.params.rigid_disp_m
    fe_data.disp.y = np.ones(fe_data.nodes.nums.shape)*fe_data.params.rigid_disp_m

    # Call function to perform general opts and save the pickle
    create_test_case(save_path,test_case,fe_data,gen_data)

    # TEST CASE 2: Small Rigid Body Motion
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'RigidBodyMotion_0_1px'
    test_case.test_val = 0.1

    # Create generic data object and unique params
    fe_data = gen_data
    fe_data.params.rigid_disp_px = test_case.test_val
    fe_data.params.rigid_disp_m = test_case.test_val*test_case.m_per_px

    # Init displacement fields
    fe_data.disp.node_nums = fe_data.nodes.nums
    fe_data.disp.x = np.ones(fe_data.nodes.nums.shape)*fe_data.params.rigid_disp_m
    fe_data.disp.y = np.ones(fe_data.nodes.nums.shape)*fe_data.params.rigid_disp_m

    # Call function to perform general opts and save the pickle
    create_test_case(save_path,test_case,fe_data,gen_data)

    # TEST CASE 3: Small hydrostatic strain
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'HydroStrain_1meta'
    test_case.test_val = -1e-3

    # Create generic data object and unique params
    fe_data = gen_data
    fe_data.params.hydro_strain = test_case.test_val

    # Init displacement fields
    fe_data.disp.node_nums = fe_data.nodes.nums

    etaH = test_case.test_val
    disp_vec_x = etaH*(node_vec_x-test_case.geom[xi]/2)
    disp_vec_y = -etaH*(node_vec_y-test_case.geom[yi]/2)
    [u_x,u_y] = np.meshgrid(disp_vec_x,disp_vec_y)

    if diag_on:
        fig, ax = plt.subplots()
        cset = plt.imshow(u_x,cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Hydro Strain: u_x',fontsize=12)
        cbar = fig.colorbar(cset)

        fig, ax = plt.subplots()
        cset = plt.imshow(u_y,cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('Hydro Strain: u_y',fontsize=12)
        cbar = fig.colorbar(cset)

    fe_data.disp.x = u_x.flatten()
    fe_data.disp.y = u_y.flatten()

    # Call function to perform general opts and save the pickle
    create_test_case(save_path,test_case,fe_data,gen_data)

    # TEST CASE 4: Large hydrostatic strain
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'HydroStrain_1pc'
    test_case.test_val = -1/100

    # Create generic data object and unique params
    fe_data = gen_data
    fe_data.params.hydro_strain = test_case.test_val

    # Init displacement fields
    fe_data.disp.node_nums = fe_data.nodes.nums

    etaH = test_case.test_val
    disp_vec_x = etaH*(node_vec_x-test_case.geom[xi]/2)
    disp_vec_y = -etaH*(node_vec_y-test_case.geom[yi]/2)
    [u_x,u_y] = np.meshgrid(disp_vec_x,disp_vec_y)

    if diag_on:
        fig, ax = plt.subplots()
        cset = plt.imshow(u_x,cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('{} , u_x'.format(test_case.test_tag),fontsize=12)
        cbar = fig.colorbar(cset)

        fig, ax = plt.subplots()
        cset = plt.imshow(u_y,cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('{} , u_y'.format(test_case.test_tag),fontsize=12)
        cbar = fig.colorbar(cset)

    fe_data.disp.x = u_x.flatten()
    fe_data.disp.y = u_y.flatten()

    # Call function to perform general opts and save the pickle
    create_test_case(save_path,test_case,fe_data,gen_data)

    # TEST CASE 5: Small Shear Strain
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'ShearStrain_1meta'
    test_case.test_val = 1e-3

    # Create generic data object and unique params
    fe_data = gen_data
    fe_data.params.shear_strain = test_case.test_val

    # Init displacement fields
    fe_data.disp.node_nums = fe_data.nodes.nums

    etaS = test_case.test_val
    u_x = etaS*(node_grid_y-test_case.geom[xi]/2)
    u_y = etaS*(node_grid_x-test_case.geom[yi]/2)

    if diag_on:
        fig, ax = plt.subplots()
        cset = plt.imshow(u_x,cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('{} , u_x'.format(test_case.test_tag),fontsize=12)
        cbar = fig.colorbar(cset)

        fig, ax = plt.subplots()
        cset = plt.imshow(u_y,cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('{} , u_y'.format(test_case.test_tag),fontsize=12)
        cbar = fig.colorbar(cset)

    fe_data.disp.x = u_x.flatten()
    fe_data.disp.y = u_y.flatten()

    # Call function to perform general opts and save the pickle
    create_test_case(save_path,test_case,fe_data,gen_data)

    # TEST CASE 6: Large Shear Strain
    test_case.test_num = test_case.test_num+1
    test_case.test_tag = 'ShearStrain_1pc'
    test_case.test_val = 1/100

    # Create generic data object and unique params
    fe_data = gen_data
    fe_data.params.shear_strain = test_case.test_val

    # Init displacement fields
    fe_data.disp.node_nums = fe_data.nodes.nums

    etaS = test_case.test_val
    u_x = etaS*(node_grid_y-test_case.geom[xi]/2)
    u_y = etaS*(node_grid_x-test_case.geom[yi]/2)

    if diag_on:
        fig, ax = plt.subplots()
        cset = plt.imshow(u_x,cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('{} , u_x'.format(test_case.test_tag),fontsize=12)
        cbar = fig.colorbar(cset)

        fig, ax = plt.subplots()
        cset = plt.imshow(u_y,cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('{} , u_y'.format(test_case.test_tag),fontsize=12)
        cbar = fig.colorbar(cset)

    fe_data.disp.x = u_x.flatten()
    fe_data.disp.y = u_y.flatten()

    # Call function to perform general opts and save the pickle
    create_test_case(save_path,test_case,fe_data,gen_data)

    # TEST CASE 0: Random Disp x10
    num_frames = 10


    rng = default_rng(42)

    test_case.test_num = 0
    test_case.test_tag = 'RandDispx10'
    test_case.test_val = 1

    # Create generic data object and unique params
    fe_data = gen_data
    fe_data.params.rand_disp_max_px = test_case.test_val
    fe_data.params.rand_disp_max_m = test_case.test_val*test_case.m_per_px
    u_max = fe_data.params.rand_disp_max_m

    # Init displacement fields
    fe_data.disp.node_nums = fe_data.nodes.nums

    u_x = (rng.random([node_grid_x.shape[0],node_grid_x.shape[1],num_frames])-0.5)*2*u_max
    u_y = (rng.random([node_grid_x.shape[0],node_grid_x.shape[1],num_frames])-0.5)*2*u_max

    if diag_on:
        fig, ax = plt.subplots()
        cset = plt.imshow(u_x[:,:,-1],cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('{} , u_x'.format(test_case.test_tag),fontsize=12)
        cbar = fig.colorbar(cset)

        fig, ax = plt.subplots()
        cset = plt.imshow(u_y[:,:,-1],cmap=plt.get_cmap(v_cmap),origin='lower')
        ax.set_aspect('equal','box')
        ax.set_title('{} , u_y'.format(test_case.test_tag),fontsize=12)
        cbar = fig.colorbar(cset)

    disp_x = np.zeros([fe_data.disp.node_nums.shape[0],num_frames])
    disp_y = np.zeros([fe_data.disp.node_nums.shape[0],num_frames])
    for ff in range(num_frames):
        disp_x[:,ff] = u_x[:,:,ff].flatten()
        disp_y[:,ff] = u_y[:,:,ff].flatten()

    fe_data.disp.x = disp_x
    fe_data.disp.y = disp_y

    # Call function to perform general opts and save the pickle
    create_test_case(save_path,test_case,fe_data,gen_data)

    # COMPLETE
    print('')
    print('--------------------------------------------------------------------')
    print('COMPLETE\n')

if __name__ == '__main__':
    main()

