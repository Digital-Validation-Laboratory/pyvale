import numpy as np
from pprint import pprint

def main() -> None:
    node_vec_x = np.arange(0,4,1)
    node_vec_y = np.arange(0,4,1)
    (node_grid_x,node_grid_y) = np.meshgrid(node_vec_x,node_vec_y)
    node_grid_y = node_grid_y[::-1,:] # flipud

    pprint(node_grid_x)
    pprint(node_grid_y)

    node_flat_x = np.atleast_2d(node_grid_x.flatten()).T
    node_flat_y = np.atleast_2d(node_grid_y.flatten()).T

    nodes = np.hstack((node_flat_x,node_flat_y))

    pprint(nodes.shape)

if __name__ == '__main__':
    main()
