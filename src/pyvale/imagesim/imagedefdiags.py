'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
import matplotlib.pyplot as plt

from pyvale.plotprops import PlotProps

PLOT_PROPS = PlotProps()
I_CMAP = 'gray'
V_CMAP = 'plasma'

def plot_diag_image(title: str,
                    image: np.ndarray,
                    cmap: str = V_CMAP) -> None:
    fig, ax = plt.subplots()
    cset = plt.imshow(image,cmap=plt.get_cmap(cmap),origin='lower')
    ax.set_aspect('equal','box')
    ax.set_title(title,fontsize=12)
    fig.colorbar(cset)


def plot_diag_image_xy(title: str,
                       image: np.ndarray,
                       extent: tuple[float,float,float,float],
                       cmap: str = V_CMAP,
                       ax_units: str = 'mm') -> None:
        fig, ax = plt.subplots()
        cset = plt.imshow(image,
                        aspect='auto',interpolation='none',
                        origin='lower',cmap=plt.get_cmap(cmap),
                        extent=extent)
        ax.set_aspect('equal','box')
        ax.set_title(title,fontsize=12)
        ax.set_xlabel(f'x [{ax_units}]',fontsize=12)
        ax.set_ylabel(f'y [{ax_units}]',fontsize=12)
        fig.colorbar(cset)