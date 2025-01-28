"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from typing import Any
import matplotlib.pyplot as plt
from pyvale.core.camera import CameraBasic2D
from pyvale.core.visualopts import PlotOptsGeneral

# NOTE: This module is a feature under developement. Linked to the basic camera
# class.

def plot_measurement_image(camera: CameraBasic2D,
                           component: str,
                           time_step: int = -1,
                           plot_opts: PlotOptsGeneral | None = None
                           ) -> tuple[Any,Any]:

    if plot_opts is None:
        plot_opts = PlotOptsGeneral()

    comp_ind = camera.get_field().get_component_index(component)
    meas_image = camera.get_measurement_images()[:,:,comp_ind,time_step]
    descriptor = camera.get_descriptor()

    (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                             layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    cset = plt.imshow(meas_image,
                      cmap=plt.get_cmap(plot_opts.cmap_seq),
                      origin='lower')
    ax.set_aspect('equal','box')

    fig.colorbar(cset,
                 label=descriptor.create_label_flat(comp_ind))

    title = f"Time: {camera.get_sample_times()[time_step]}s"
    ax.set_title(title,fontsize=plot_opts.font_head_size)
    ax.set_xlabel(r"x ($px$)",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(r"y ($px$)",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    return (fig,ax)
