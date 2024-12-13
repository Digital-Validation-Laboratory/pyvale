import numpy as np
from scipy.spatial.transform import Rotation

def main() -> None:
    pos = np.array([[-1,-1,0],
                    [1,-1,0],
                    [1,1,0],
                    [-1,1,0]])
    print(pos)

    rot = Rotation.from_euler("zyx",[90,0,0],degrees=True)
    rot_mat = rot.as_matrix()
    print(rot_mat)

    print(np.matmul(rot_mat,pos.T).T)

if __name__ == "__main__":
    main()