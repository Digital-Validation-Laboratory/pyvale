'''
================================================================================
pycave: plotprops

authors: thescepticalrabbit
================================================================================
'''
from dataclasses import dataclass

@dataclass
class PlotProps:
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
