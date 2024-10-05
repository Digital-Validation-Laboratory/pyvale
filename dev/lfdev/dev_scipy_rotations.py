from scipy.spatial.transform import Rotation as R
import numpy as np

def main() -> None:
    v0 = np.array(((1.0,0.0),
                   (1.0,0.0),
                   (1.0,0.0),
                   (1.0,0.0)))

    r0 = R.identity()

    # intrinsic, refers to last rotated coords in order
    r1 = R.from_euler("ZYX", [90, 0, 0], degrees=True)

    # extrinsic, refer to fixed global coords
    r2 = R.from_euler("zyx", [90, 0, 0], degrees=True)

    v0_r = r2.apply(v0)
    print(v0)
    print(v0_r)


if __name__ == '__main__':
    main()