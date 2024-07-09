import numpy as np

def main() -> None:
    dims = np.atleast_2d(np.array([1.0,2.0,0.0]))
    gauss_pt_offsets = dims * 1/np.sqrt(3)* np.array([[-1,-1,0],
                                                        [-1,1,0],
                                                        [1,-1,0],
                                                        [1,1,0]])
    print(gauss_pt_offsets)

if __name__ == "__main__":
    main()

