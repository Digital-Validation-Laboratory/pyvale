from scipy.spatial.transform import Rotation as R
import numpy as np

def main() -> None:
    v0 = np.array((1,0,0))
    v1 = np.array((1,1,1))

    print(v0)
    print(v1)

    r0 = R.identity()
    r1 = R.from_euler("ZYX", [90, 0, 0], degrees=True)  # intrinsic
    r2 = R.from_euler("zyx", [90, 0, 0], degrees=True)  # extrinsic

    v0_r = r2.apply(v0)
    print(v0)
    print(v0_r)


if __name__ == '__main__':
    main()