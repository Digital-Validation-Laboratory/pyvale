import numpy as np
from pprint import pprint


def main() -> None:
    v = np.zeros((3,2))
    v[0,1] = 1
    v[1,0] = 1
    v[2,0] = 1
    v[2,1] = 1

    n1 = np.linalg.norm(v,axis=1)
    print(n1)

    n1 = np.linalg.norm(v,axis=0)
    print(n1)

if __name__ == '__main__':
    main()
