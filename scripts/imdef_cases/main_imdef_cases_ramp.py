# -*- coding: utf-8 -*-
"""
GENERATE IMAGE DEFORMATION TEST CASES
Created on Wed Dec  8 08:46:13 2021

@author: Lloyd Fletcher
"""

# IMPORTS
import os
import time
from pprint import pprint
import pickle
from pathlib import Path

import numpy as np

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
        self.elem_size = [0.5e-3,0.5e-3]
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
    # NOTE: arange doesn't include end value!
    node_vec_x = np.arange(0,test_case.geom[xi]+test_case.elem_size[xi],test_case.elem_size[xi])
    node_vec_y = np.arange(0,test_case.geom[yi]+test_case.elem_size[yi],test_case.elem_size[yi])
    (node_grid_x,node_grid_y) = np.meshgrid(node_vec_x,node_vec_y)
    node_grid_y = node_grid_y[::-1,:] # flipud

    print('Creating empty FE data object.')
    gen_data = ar.FEData(cwd,0,2)

    gen_data.nodes.loc_x = node_grid_x.flatten()
    gen_data.nodes.loc_y = node_grid_y.flatten()
    gen_data.nodes.loc_z = np.zeros(gen_data.nodes.loc_x.shape)
    gen_data.nodes.nums = np.arange(1,len(gen_data.nodes.loc_x)+1)


    # TEST CASE 1: Rigid Body Motion
    # Linear ramp up to 1px of rigid body motion in X and Y
    test_case.test_num = 7
    test_case.test_tag = 'RampRigidBodyMotion_1_0px'
    test_case.test_val = 1.0
    num_frames = 10

    # Create generic data object and unique params
    fe_data = gen_data
    fe_data.params.rigid_disp_px_max = test_case.test_val
    fe_data.params.rigid_disp_m_max = test_case.test_val*test_case.m_per_px

    # Create a vector of rigid displacements
    disp_max_m = fe_data.params.rigid_disp_m_max
    disp_inc_m = disp_max_m/num_frames
    disp_vec_m = np.arange(disp_inc_m,disp_max_m+disp_inc_m,disp_inc_m)

    # Init displacement fields
    fe_data.disp.node_nums = fe_data.nodes.nums

    disp_x = np.zeros([fe_data.disp.node_nums.shape[0],num_frames])
    disp_y = np.zeros([fe_data.disp.node_nums.shape[0],num_frames])
    for ff in range(num_frames):
        disp_x[:,ff] = disp_vec_m[ff]
        disp_y[:,ff] = disp_vec_m[ff]

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

