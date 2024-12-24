"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np

def transform_vector_2d(trans_mat: np.ndarray, vector: np.ndarray
                        ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans_mat : np.ndarray
        _description_
    vector : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    vector_trans = np.zeros_like(vector)
    (xx,yy) = (0,1)

    vector_trans[xx,:] = (trans_mat[0,0]*vector[xx,:]
                        + trans_mat[0,1]*vector[yy,:])
    vector_trans[yy,:] = (trans_mat[0,1]*vector[xx,:]
                        + trans_mat[1,1]*vector[yy,:])
    return vector_trans


def transform_vector_3d(trans_mat: np.ndarray, vector: np.ndarray
                       ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans_mat : np.ndarray
        _description_
    vector : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    vector_trans = np.zeros_like(vector)
    (xx,yy,zz) = (0,1,2)

    vector_trans[xx,:] = (trans_mat[0,0]*vector[xx,:]
                          + trans_mat[0,1]*vector[yy,:]
                          + trans_mat[0,2]*vector[zz,:])
    vector_trans[yy,:] = (trans_mat[0,1]*vector[xx,:]
                          + trans_mat[1,1]*vector[yy,:]
                          + trans_mat[1,2]*vector[zz,:])
    vector_trans[zz,:] = (trans_mat[0,2]*vector[xx,:]
                          + trans_mat[1,2]*vector[yy,:]
                          + trans_mat[2,2]*vector[zz,:])

    return vector_trans

def transform_vector_2d_batch(trans_mat: np.ndarray, vector: np.ndarray
                        ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans_mat : np.ndarray
        _description_
    vector : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    vector_trans = np.zeros_like(vector)
    (xx,yy) = (0,1)

    vector_trans[:,xx,:] = (trans_mat[0,0]*vector[:,xx,:]
                            + trans_mat[0,1]*vector[:,yy,:])
    vector_trans[:,yy,:] = (trans_mat[0,1]*vector[:,xx,:]
                            + trans_mat[1,1]*vector[:,yy,:])

    return vector_trans


def transform_vector_3d_batch(trans_mat: np.ndarray, vector: np.ndarray
                             ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans_mat : np.ndarray
        _description_
    vector : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    vector_trans = np.zeros_like(vector)
    (xx,yy,zz) = (0,1,2)

    vector_trans[:,xx,:] = (trans_mat[0,0]*vector[:,xx,:]
                            + trans_mat[0,1]*vector[:,yy,:]
                            + trans_mat[0,2]*vector[:,zz,:])
    vector_trans[:,yy,:] = (trans_mat[0,1]*vector[:,xx,:]
                            + trans_mat[1,1]*vector[:,yy,:]
                            + trans_mat[1,2]*vector[:,zz,:])
    vector_trans[:,zz,:] = (trans_mat[0,2]*vector[:,xx,:]
                            + trans_mat[1,2]*vector[:,yy,:]
                            + trans_mat[2,2]*vector[:,zz,:])

    return vector_trans

def transform_tensor_2d(trans_mat: np.ndarray, tensor: np.ndarray
                        ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans_mat : np.ndarray
        _description_
    tensor : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    tensor_trans = np.zeros_like(tensor)
    (xx,yy,xy) = (0,1,2)

    tensor_trans[xx,:] = (trans_mat[0,0]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:])
                        + trans_mat[0,1]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]))

    tensor_trans[yy,:] = (trans_mat[0,1]*(trans_mat[0,1]*tensor[xx,:]
                                          + trans_mat[1,1]*tensor[xy,:])
                        + trans_mat[1,1]*(trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[1,1]*tensor[yy,:]))

    tensor_trans[xy,:] = (trans_mat[0,1]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:])
                        + trans_mat[1,1]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]))

    return tensor_trans


def transform_tensor_3d(trans_mat: np.ndarray, tensor: np.ndarray
                       ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans_mat : np.ndarray
        _description_
    tensor : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    tensor_trans = np.zeros_like(tensor)
    (xx,yy,zz,xy,xz,yz) = (0,1,2,3,4,5)

    tensor_trans[xx,:] = (trans_mat[0,0]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[0,2]*tensor[xz,:])
                        + trans_mat[0,1]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]
                                          + trans_mat[0,2]*tensor[yz,:])
                        + trans_mat[0,2]*(trans_mat[0,0]*tensor[xz,:]
                                          + trans_mat[0,1]*tensor[yz,:]
                                          + trans_mat[0,2]*tensor[zz,:]))

    tensor_trans[yy,:] = (trans_mat[0,1]*(trans_mat[0,1]*tensor[xx,:]
                                          + trans_mat[1,1]*tensor[xy,:]
                                          + trans_mat[1,2]*tensor[xz,:])
                        + trans_mat[1,1]*(trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[1,1]*tensor[yy,:]
                                          + trans_mat[1,2]*tensor[yz,:])
                        + trans_mat[1,2]*(trans_mat[0,1]*tensor[xz,:]
                                          + trans_mat[1,1]*tensor[yz,:]
                                          + trans_mat[1,2]*tensor[zz,:]))

    tensor_trans[zz,:] = (trans_mat[0,2]*(trans_mat[0,2]*tensor[xx,:]
                                          + trans_mat[1,2]*tensor[xy,:]
                                          + trans_mat[2,2]*tensor[xz,:])
                        + trans_mat[1,2]*(trans_mat[0,2]*tensor[xy,:]
                                          + trans_mat[1,2]*tensor[yy,:]
                                          + trans_mat[2,2]*tensor[yz,:])
                        + trans_mat[2,2]*(trans_mat[0,2]*tensor[xz,:]
                                          + trans_mat[1,2]*tensor[yz,:]
                                          + trans_mat[2,2]*tensor[zz,:]))

    tensor_trans[xy,:] = (trans_mat[0,1]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[0,2]*tensor[xz,:])
                        + trans_mat[1,1]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]
                                          + trans_mat[0,2]*tensor[yz,:])
                        + trans_mat[1,2]*(trans_mat[0,0]*tensor[xz,:]
                                          + trans_mat[0,1]*tensor[yz,:]
                                          + trans_mat[0,2]*tensor[zz,:]))

    tensor_trans[xz,:] = (trans_mat[0,2]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[0,2]*tensor[xz,:])
                        + trans_mat[1,2]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]
                                          + trans_mat[0,2]*tensor[yz,:])
                        + trans_mat[2,2]*(trans_mat[0,0]*tensor[xz,:]
                                          + trans_mat[0,1]*tensor[yz,:]
                                          + trans_mat[0,2]*tensor[zz,:]))

    tensor_trans[yz,:] = (trans_mat[0,2]*(trans_mat[0,1]*tensor[xx,:]
                                         + trans_mat[1,1]*tensor[xy,:]
                                         + trans_mat[1,2]*tensor[xz,:])
                        + trans_mat[1,2]*(trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[1,1]*tensor[yy,:]
                                          + trans_mat[1,2]*tensor[yz,:])
                        + trans_mat[2,2]*(trans_mat[0,1]*tensor[xz,:]
                                          + trans_mat[1,1]*tensor[yz,:]
                                          + trans_mat[1,2]*tensor[zz,:]))

    return tensor_trans


def transform_tensor_2d_batch(trans_mat: np.ndarray, tensor: np.ndarray
                             ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans_mat : np.ndarray
        _description_
    tensor : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    tensor_trans = np.zeros_like(tensor)
    (xx,yy,xy) = (0,1,2)

    tensor_trans[:,xx,:] = (trans_mat[0,0]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:])
                          + trans_mat[0,1]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]))

    tensor_trans[:,yy,:] = (trans_mat[0,1]*(trans_mat[0,1]*tensor[:,xx,:]
                                            + trans_mat[1,1]*tensor[:,xy,:])
                          + trans_mat[1,1]*(trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[1,1]*tensor[:,yy,:]))

    tensor_trans[:,xy,:] = (trans_mat[0,1]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:])
                          + trans_mat[1,1]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]))

    return tensor_trans


def transform_tensor_3d_batch(trans_mat: np.ndarray, tensor: np.ndarray
                              ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    trans_mat : np.ndarray
        _description_
    tensor : np.ndarray
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    tensor_trans = np.zeros_like(tensor)
    (xx,yy,zz,xy,xz,yz) = (0,1,2,3,4,5)

    tensor_trans[:,xx,:] = (trans_mat[0,0]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[0,2]*tensor[:,xz,:])
                          + trans_mat[0,1]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]
                                            + trans_mat[0,2]*tensor[:,yz,:])
                          + trans_mat[0,2]*(trans_mat[0,0]*tensor[:,xz,:]
                                            + trans_mat[0,1]*tensor[:,yz,:]
                                            + trans_mat[0,2]*tensor[:,zz,:]))

    tensor_trans[:,yy,:] = (trans_mat[0,1]*(trans_mat[0,1]*tensor[:,xx,:]
                                            + trans_mat[1,1]*tensor[:,xy,:]
                                            + trans_mat[1,2]*tensor[:,xz,:])
                          + trans_mat[1,1]*(trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[1,1]*tensor[:,yy,:]
                                            + trans_mat[1,2]*tensor[:,yz,:])
                          + trans_mat[1,2]*(trans_mat[0,1]*tensor[:,xz,:]
                                            + trans_mat[1,1]*tensor[:,yz,:]
                                            + trans_mat[1,2]*tensor[:,zz,:]))

    tensor_trans[:,zz,:] = (trans_mat[0,2]*(trans_mat[0,2]*tensor[:,xx,:]
                                            + trans_mat[1,2]*tensor[:,xy,:]
                                            + trans_mat[2,2]*tensor[:,xz,:])
                          + trans_mat[1,2]*(trans_mat[0,2]*tensor[:,xy,:]
                                            + trans_mat[1,2]*tensor[:,yy,:]
                                            + trans_mat[2,2]*tensor[:,yz,:])
                          + trans_mat[2,2]*(trans_mat[0,2]*tensor[:,xz,:]
                                            + trans_mat[1,2]*tensor[:,yz,:]
                                            + trans_mat[2,2]*tensor[:,zz,:]))

    tensor_trans[:,xy,:] = (trans_mat[0,1]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[0,2]*tensor[:,xz,:])
                          + trans_mat[1,1]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]
                                            + trans_mat[0,2]*tensor[:,yz,:])
                          + trans_mat[1,2]*(trans_mat[0,0]*tensor[:,xz,:]
                                            + trans_mat[0,1]*tensor[:,yz,:]
                                            + trans_mat[0,2]*tensor[:,zz,:]))

    tensor_trans[:,xz,:] = (trans_mat[0,2]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[0,2]*tensor[:,xz,:])
                          + trans_mat[1,2]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]
                                            + trans_mat[0,2]*tensor[:,yz,:])
                          + trans_mat[2,2]*(trans_mat[0,0]*tensor[:,xz,:]
                                            + trans_mat[0,1]*tensor[:,yz,:]
                                            + trans_mat[0,2]*tensor[:,zz,:]))

    tensor_trans[:,yz,:] = (trans_mat[0,2]*(trans_mat[0,1]*tensor[:,xx,:]
                                            + trans_mat[1,1]*tensor[:,xy,:]
                                            + trans_mat[1,2]*tensor[:,xz,:])
                          + trans_mat[1,2]*(trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[1,1]*tensor[:,yy,:]
                                            + trans_mat[1,2]*tensor[:,yz,:])
                          + trans_mat[2,2]*(trans_mat[0,1]*tensor[:,xz,:]
                                            + trans_mat[1,1]*tensor[:,yz,:]
                                            + trans_mat[1,2]*tensor[:,zz,:]))

    return tensor_trans