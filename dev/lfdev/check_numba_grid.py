import numpy as np
import numba

@numba.jit(nopython=True)
def meshgrid2(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    x_grid = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    y_grid = np.empty(shape=(y.size, x.size), dtype=y.dtype)

    for ii in range(y.size):
        for jj in range(x.size):
            x_grid[ii,jj] = x[jj]
            y_grid[ii,jj] = y[ii]

    return (x_grid,y_grid)

def main() -> None:
    x = np.arange(0,6,0.5)
    y = np.arange(0,2,0.25)
    (x_grid,y_grid) = np.meshgrid(x,y)

    print(x_grid)
    print(y_grid)

    (x_grid2,y_grid2) = meshgrid2(x,y)
    print()
    print(x_grid2)
    print(y_grid2)


if __name__ == "__main__":
    main()