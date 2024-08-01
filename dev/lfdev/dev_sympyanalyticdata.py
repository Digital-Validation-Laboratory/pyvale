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
    case_data.num_elem_x = 4
    case_data.num_elem_y = 3
    case_data.time_steps = np.linspace(0.0,1.0,11)
    case_data.field_keys = ('temperature',)

    sym_x = sympy.Symbol("x")
    sym_y = sympy.Symbol("y")
    sym_t = sympy.Symbol("t")
    case_data.funcs_x = (sym_x*(sym_x - case_data.length_x),)
    case_data.funcs_y = (sym_y*(sym_y - case_data.length_y),)
    case_data.funcs_t = (1.0,)
    #===========================================================================

    data_gen = pyvale.AnalyticSimDataGenerator(case_data)
    sim_data = data_gen.generate_sim_data()

    (full_x,full_y,full_time) = pyvale.fill_dims(sim_data.coords[:,0],
                                                 sim_data.coords[:,1],
                                                 sim_data.time)

    print()
    print(f'{full_x.shape=}')
    print(f'{full_y.shape=}')
    print(f'{full_time.shape=}')

    print(full_x)
    print(full_y)

    print(sim_data.node_vars['temperature'][:,-1])

if __name__ == '__main__':
    main()
