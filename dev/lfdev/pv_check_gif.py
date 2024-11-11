from __future__ import annotations
import numpy as np
import pyvista as pv

def main() -> None:
    x = np.arange(-10, 10, 0.5)
    y = np.arange(-10, 10, 0.5)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    z = np.sin(r)

    # Create and structured surface
    grid = pv.StructuredGrid(x, y, z)
    grid["Height"] = z.ravel()
    #grid.plot()

    # Create a plotter object and set the scalars to the Z height
    plotter = pv.Plotter(off_screen=False)

    plotter.add_mesh(
        grid,
        scalars="Height",
        lighting=False,
        show_edges=True,
        clim=[-1, 1],
    )

    # Open a gif
    #plotter.open_gif("wave.gif")

    # Update Z and write a frame for each updated position
    nframe = 30

    plotter.show(auto_close=False,interactive=False)

    for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
        z = np.sin(r + phase)
        # Update values inplace
        grid.points[:, -1] = z.ravel()
        grid["Height"] = z.ravel()
        # Write a frame. This triggers a render.
        plotter.render()
        #plotter.write_frame()

    # Closes and finalizes movie
    plotter.close()

if __name__ == "__main__":
    main()