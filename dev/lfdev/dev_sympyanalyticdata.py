'''
================================================================================
Analytic test case data - linear

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import sympy
import scipy.integrate
import mooseherder as mh
import pyvale
import pyvale.visualisation.plotters

def main() -> None:

    #===========================================================================
    # User defined test case parameters
    case_data = pyvale.AnalyticCaseData2D()

    case_data.length_x = 10
    case_data.length_y = 7.5
    case_data.num_elem_x = 4*10
    case_data.num_elem_y = 3*10
    case_data.time_steps = np.linspace(0.0,1.0,11)
    case_data.field_keys = ('temperature',)

    sym_x = sympy.Symbol("x")
    sym_y = sympy.Symbol("y")
    sym_t = sympy.Symbol("t")
    #case_data.funcs_x = (sym_x*(sym_x - case_data.length_x),)
    #case_data.funcs_y = (sym_y*(sym_y - case_data.length_y),)
    case_data.funcs_x = (50.0*sympy.sin( 2*sympy.pi/(case_data.length_x/10) * sym_x)+50,)
    case_data.funcs_y = (1.0,)
    case_data.funcs_t = (1.0,)
    #===========================================================================

    data_gen = pyvale.AnalyticSimDataGenerator(case_data)
    sim_data = data_gen.generate_sim_data()
    (grid_x,grid_y,grid_field) = data_gen.get_visualisation_grid()

    fig, ax = plt.subplots()
    cs = ax.contourf(grid_x,
                     grid_y,
                     grid_field)
    cbar = fig.colorbar(cs)
    plt.axis('scaled')
    #plt.show()

    (sym_y,sym_x,sym_t) = sympy.symbols("y,x,t")




if __name__ == '__main__':
    main()
