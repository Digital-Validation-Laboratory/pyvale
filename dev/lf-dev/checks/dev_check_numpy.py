import numpy as np
from pprint import pprint

def main() -> None:
    node_vec_x = np.arange(0,4,1)
    node_vec_y = np.arange(0,4,1)


    node_flat_x = np.atleast_2d(node_vec_x)
    node_flat_y = np.atleast_2d(node_vec_y)

    pprint(node_flat_x.shape)

if __name__ == '__main__':
    main()
