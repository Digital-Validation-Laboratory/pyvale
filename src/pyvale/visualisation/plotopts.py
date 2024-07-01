'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from dataclasses import dataclass
import numpy as np

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

    single_fig_scale: float = 0.65
    single_fig_size: tuple[float,float] = (a4_print_width*single_fig_scale,
                        a4_print_width*single_fig_scale/aspect_ratio)

    resolution: int = 300

    font_name: str = 'Liberation Sans'
    font_def_weight: str = 'normal'
    font_def_size: float = 10
    font_tick_size: float = 9
    font_head_size: float = 12
    font_ax_size: float = 11
    font_leg_size: float = 9

    ms: float = 6.0
    lw: float = 1.0

    cmap_seq: str = "cividis"
    cmap_div: str = "RdBu"


@dataclass
class SensorPlotOpts:
    legend: bool = True
    meas_label: str = r'Measured Value, $M_{s}$ [unit]'
    x_label: str = r'x [m]'
    y_label: str = r'y [m]'
    z_label: str = r'z [m]'
    time_label: str = r'Time, $t$ [s]'

    truth_line: str | None = '-'
    sim_line: str | None = '-'
    meas_line: str = '--+'

    sensor_tag: str = 'S'
    sensors_to_plot: np.ndarray | None = None
    time_inds: np.ndarray | None = None


def create_label_str(descriptors: tuple[str,str,str,str]) -> str:
    return rf"{descriptors[0]}, {descriptors[1]} [{descriptors[2]}]"


def create_sensor_tags(tag: str, n_sensors: int) -> list[str]:
    z_width = int(np.log10(n_sensors))+1

    sensor_names = list()
    for ss in range(n_sensors):
        num_str = f'{ss}'.zfill(z_width)
        sensor_names.append(f'{tag}{num_str}')

    return sensor_names