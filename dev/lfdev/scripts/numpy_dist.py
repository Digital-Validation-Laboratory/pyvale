import numpy as np

def main() -> None:
    x_vec = np.linspace(0,9,10)+0.5
    y_vec = np.linspace(0,9,10)+0.5
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)
    x_px = np.atleast_2d(x_grid.flatten())
    y_px = np.atleast_2d(y_grid.flatten())


    dots_x_vec = np.array([2.5,7.5])
    dots_y_vec = np.array([2.5,7.5])
    dot_rad = 2
    (dots_x,dots_y) = np.meshgrid(dots_x_vec,dots_y_vec)
    dots_x = np.atleast_2d(dots_x.flatten())
    dots_y = np.atleast_2d(dots_y.flatten())

    num_dots = np.max(dots_x.shape)
    num_px = np.max(x_px.shape)

    dots_x = np.repeat(dots_x,num_px,axis=0)
    dots_y = np.repeat(dots_y,num_px,axis=0)

    x_px = np.repeat(x_px.T,num_dots,axis=1)
    y_px = np.repeat(y_px.T,num_dots,axis=1)

    dist = np.sqrt((dots_x-x_px)**2 + (dots_y-y_px)**2)
    image = np.zeros_like(dist)
    image[dist < dot_rad] = 1
    image = np.sum(image,axis=1)

    image = np.reshape(image,x_grid.shape)

    print(f"{num_px=}")
    print(f"{num_dots=}")

    print(f"{dots_x.shape=}")
    print(f"{x_px.shape=}")
    print(dist < dot_rad)
    print(f"{image.shape=}")
    print()
    print(image)



if __name__ == "__main__":
    main()