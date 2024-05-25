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

def plot_diag_image(title: str, image: np.ndarray, cmap: str) -> None:
    fig, ax = plt.subplots()
    cset = plt.imshow(image,cmap=plt.get_cmap(cmap),origin='lower')
    ax.set_aspect('equal','box')
    ax.set_title(title,fontsize=12)
    fig.colorbar(cset)