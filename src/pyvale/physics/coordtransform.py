'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
import numpy as np

def transform_tensor_2d(rot_mat: np.ndarray, tensor: np.ndarray
                        ) -> np.ndarray:

    tensor_rot = np.zeros_like(tensor)
    (xx,yy,xy) = (0,1,2)

    tensor_rot[xx,:] = rot_mat[0,0]*(rot_mat[0,0]*tensor[xx,:] + rot_mat[0,1]*tensor[xy,:]) + \
                       rot_mat[0,1]*(rot_mat[0,0]*tensor[xy,:] + rot_mat[0,1]*tensor[yy,:])

    tensor_rot[yy,:] = rot_mat[0,1]*(rot_mat[0,1]*tensor[xx,:] + rot_mat[1,1]*tensor[xy,:]) + \
                       rot_mat[1,1]*(rot_mat[0,1]*tensor[xy,:] + rot_mat[1,1]*tensor[yy,:])

    tensor_rot[xy,:] = rot_mat[0,1]*(rot_mat[0,0]*tensor[xx,:] + rot_mat[0,1]*tensor[xy,:]) + \
                       rot_mat[1,1]*(rot_mat[0,0]*tensor[xy,:] + rot_mat[0,1]*tensor[yy,:])

    return tensor_rot


def transform_tensor_3d(rot_mat: np.ndarray, tensor: np.ndarray
                        ) -> np.ndarray:

    tensor_rot = np.zeros_like(tensor)
    (xx,yy,zz,xy,xz,yz) = (0,1,2,3,4,5,6)

    tensor_rot[xx,:] = rot_mat[0,0]*(rot_mat[0,0]*tensor[xx,:] + rot_mat[0,1]*tensor[xy,:] + rot_mat[0,2]*tensor[xz,:]) + \
                      rot_mat[0,1]*(rot_mat[0,0]*tensor[xy,:] + rot_mat[0,1]*tensor[yy,:] + rot_mat[0,2]*tensor[yz,:]) + \
                      rot_mat[0,2]*(rot_mat[0,0]*tensor[xz,:] + rot_mat[0,1]*tensor[yz,:] + rot_mat[0,2]*tensor[zz,:])

    tensor_rot[yy,:] = rot_mat[0,1]*(rot_mat[0,1]*tensor[xx,:] + rot_mat[1,1]*tensor[xy,:] + rot_mat[1,2]*tensor[xz,:]) + \
                      rot_mat[1,1]*(rot_mat[0,1]*tensor[xy,:] + rot_mat[1,1]*tensor[yy,:] + rot_mat[1,2]*tensor[yz,:]) + \
                      rot_mat[1,2]*(rot_mat[0,1]*tensor[xz,:] + rot_mat[1,1]*tensor[yz,:] + rot_mat[1,2]*tensor[zz,:])

    tensor_rot[zz,:] = rot_mat[0,2]*(rot_mat[0,2]*tensor[xx,:] + rot_mat[1,2]*tensor[xy,:] + rot_mat[2,2]*tensor[xz,:]) + \
                      rot_mat[1,2]*(rot_mat[0,2]*tensor[xy,:] + rot_mat[1,2]*tensor[yy,:] + rot_mat[2,2]*tensor[yz,:]) + \
                      rot_mat[2,2]*(rot_mat[0,2]*tensor[xz,:] + rot_mat[1,2]*tensor[yz,:] + rot_mat[2,2]*tensor[zz,:])

    tensor_rot[xy,:] = rot_mat[0,1]*(rot_mat[0,0]*tensor[xx,:] + rot_mat[0,1]*tensor[xy,:] + rot_mat[0,2]*tensor[xz,:]) + \
                      rot_mat[1,1]*(rot_mat[0,0]*tensor[xy,:] + rot_mat[0,1]*tensor[yy,:] + rot_mat[0,2]*tensor[yz,:]) + \
                      rot_mat[1,2]*(rot_mat[0,0]*tensor[xz,:] + rot_mat[0,1]*tensor[yz,:] + rot_mat[0,2]*tensor[zz,:])

    tensor_rot[xz,:] = rot_mat[0,2]*(rot_mat[0,0]*tensor[xx,:] + rot_mat[0,1]*tensor[xy,:] + rot_mat[0,2]*tensor[xz,:]) + \
                      rot_mat[1,2]*(rot_mat[0,0]*tensor[xy,:] + rot_mat[0,1]*tensor[yy,:] + rot_mat[0,2]*tensor[yz,:]) + \
                      rot_mat[2,2]*(rot_mat[0,0]*tensor[xz,:] + rot_mat[0,1]*tensor[yz,:] + rot_mat[0,2]*tensor[zz,:])

    tensor_rot[yz,:] = rot_mat[0,2]*(rot_mat[0,1]*tensor[xx,:] + rot_mat[1,1]*tensor[xy,:] + rot_mat[1,2]*tensor[xz,:]) + \
                      rot_mat[1,2]*(rot_mat[0,1]*tensor[xy,:] + rot_mat[1,1]*tensor[yy,:] + rot_mat[1,2]*tensor[yz,:]) + \
                      rot_mat[2,2]*(rot_mat[0,1]*tensor[xz,:] + rot_mat[1,1]*tensor[yz,:] + rot_mat[1,2]*tensor[zz,:])

    return tensor_rot