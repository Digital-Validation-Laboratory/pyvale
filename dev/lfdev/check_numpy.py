import numpy as np

def main() -> None:
    v0 = np.array([[0,1,0],
                   [1,0,0]])
    v0 = v0.T
    v1 = np.array([[1,0,0],
                   [0,1,0]])
    v1 = v1.T

    print()
    print(f"{v0.shape=}")
    print(f"{v1.shape=}")
    print()
    print(np.cross(v0,v1,axisa=0,axisb=0))
    print()

if __name__ == "__main__":
    main()