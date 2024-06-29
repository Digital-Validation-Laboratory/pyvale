# https://github.com/pyvista/pyvista/discussions/2184#discussioncomment-2164786

import netCDF4
import numpy as np
import pyvista as pv
import vtk

##############################################################################
# read n'th step data and write into PyVista UnstructuredGrid
# By Yan Zhan (2022)
# Input: fname = string (file name)
#        nstep (int):  the time step
# output: PyVista UnstructuredGrid Data
def exodus2PyVista(filename, nstep=1):
    # read exodus by netCFD4
    model = netCDF4.Dataset(filename)
    # read coordinates
    X_all = np.ma.getdata(model.variables['coordx'][:])
    Y_all = np.ma.getdata(model.variables['coordy'][:])
    Z_all = np.ma.getdata(model.variables['coordz'][:])
    # ensemble the points
    points = np.vstack([X_all,Y_all,Z_all]).T
    # how node is mapped
    elem_node = np.ma.getdata(model.variables['connect1'][:])-1

    # create PyVista UnstructuredGrid
    grid = pv.UnstructuredGrid({vtk.VTK_TETRA: elem_node}, points)

    # get the name of the variables
    name_nod_var = getNames(model,'name_nod_var') #nodal data
    name_elem_var = getNames(model,'name_elem_var') #element data
    # write the data in the PyVista mesh (nodal)
    for i, nnv in enumerate(name_nod_var):
        grid[nnv] = model.variables['vals_nod_var{}'.format(i+1)][:][nstep]
    # write the data in the PyVista mesh (element)
    for i, nev in enumerate(name_elem_var):
        grid[nev] = model.variables['vals_elem_var{}eb1'.format(i+1)][:][nstep]
    # close the model
    model.close()

    return grid

##############################################################################
# Get Names in a catalog (key)
# By Yan Zhan (2022)
def getNames(model, key='name_nod_var'):
    # name of the element variables
    name_var = []
    for vname in np.ma.getdata(model.variables[key][:]).astype('U8'):
        name_var.append(''.join(vname))
    return name_var