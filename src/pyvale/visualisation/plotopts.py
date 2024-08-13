'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from dataclasses import dataclass
import numpy as np
import matplotlib as plt

@dataclass
class GeneralPlotOpts:
    """ Helper class to set properties in matplotlib for scaling to use in a
    journal article or report.
    """
    aspect_ratio: float = 1.62
    a4_width: float = 8.25
    a4_height: float = 11.75
    a4_margin_width: float = 1
    a4_margin_height: float = 1
    a4_print_width: float = a4_width-2*a4_margin_width
    a4_print_height: float = a4_height-2*a4_margin_height

    single_fig_scale: float = 0.75
    single_fig_size: tuple[float,float] = (a4_print_width*single_fig_scale,
                        a4_print_width*single_fig_scale/aspect_ratio)

    resolution: int = 200

    font_name: str = 'Liberation Sans'
    font_def_weight: str = 'normal'
    font_def_size: float = 10
    font_tick_size: float = 9
    font_head_size: float = 10
    font_ax_size: float = 10
    font_leg_size: float = 9

    ms: float = 2.4
    lw: float = 0.8

    cmap_seq: str = "cividis"
    cmap_div: str = "RdBu"

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_colors = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])


@dataclass
class SensorTraceOpts:
    legend: bool = True

    x_label: str = r'x [$m$]'
    y_label: str = r'y [$m$]'
    z_label: str = r'z [$m$]'
    time_label: str = r'Time, $t$ [$s$]'

    truth_line: str | None = '-'
    sim_line: str | None = '-'
    meas_line: str = '--o'

    sensors_to_plot: np.ndarray | None = None
    time_min_max: tuple[float,float] | None = None

