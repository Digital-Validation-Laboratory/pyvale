import numpy as np

def main() -> None:
    n_px = 10
    ss = 2
    px = np.arange(0,n_px)
    subpx = np.arange(0,2*n_px)

    start = 2
    end = 3
    check1 = subpx[2*start:2*end]

    print()
    print(px)
    print(subpx)
    print()
    print(check1)

if __name__ == "__main__":
    main()