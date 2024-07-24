'''
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np


def main() -> None:
    leng_x = 100
    leng_y = 75
    n_elem_x = 4
    n_elem_y = 3
    n_elems = n_elem_x*n_elem_y
    l_elem_x = leng_x/n_elem_x
    l_elem_y = leng_y/n_elem_y

    n_node_x = n_elem_x+1
    n_node_y = n_elem_y+1
    n_nodes = n_node_x*n_node_y

    






if __name__ == '__main__':
    main()
