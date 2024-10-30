from scipy.spatial.transform import Rotation as R
import numpy as np

def main() -> None:
    v0 = np.array(((1.0,0.0,0.0),
                   (0.0,1.0,0.0),
                   (1.0,0.0,0.0),
                   (0.0,1.0,0.0),
                   (0.0,0.0,1.0)))
    v0 = v0.T

    r0 = R.from_euler("zyx", [30, 0, 0], degrees=True)
    m0 = r0.as_matrix()

    v0_r = r0.apply(v0.T)
    print(v0)
    print(v0_r)

    # NOTE:
    # Rotation = object rotates coords fixed, sin neg row 1
    # Transformation = coords rotate object fixed, win neg row 2, transpose scipy
    print()
    print(m0.T)
    print()

    print(f"{m0.shape} x {v0.shape}")

    print(np.matmul(m0,v0).shape)
    print(np.matmul(m0,v0))
    print()


if __name__ == '__main__':
    main()