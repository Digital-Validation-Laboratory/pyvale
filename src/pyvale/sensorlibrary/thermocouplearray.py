'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Callable, Any
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from pyvale.field import Field
from pyvale.sensorarray import SensorArray, MeasurementData
from pyvale.plotprops import PlotProps
from pyvale.syserrintegrator import SysErrIntegrator
from pyvale.randerrintegrator import RandErrIntegrator


class ThermocoupleArray(SensorArray):
    def __init__(self,
                 positions: np.ndarray,
                 field: Field,
                 sample_times: np.ndarray | None = None
                 ) -> None:

        self._positions = positions
        self._field = field
        self._sample_times = sample_times

        self._sys_err_int = None
        self._rand_err_int = None

        self._sensor_names = list([])
        for ss in range(self.get_num_sensors()):
            num_str = f'{ss}'.zfill(2)
            self._sensor_names.append(f'TC{num_str}')

    #---------------------------------------------------------------------------
    # Basic getters / setters
    def get_positions(self) -> np.ndarray:
        return self._positions

    def get_sample_times(self) -> np.ndarray:
        if self._sample_times is None:
            return self._field.get_time_steps()

        return self._sample_times

    def get_num_sensors(self) -> int:
        return self._positions.shape[0]

    def get_measurement_shape(self) -> tuple[int,int,int]:
        return (self.get_num_sensors(),
                len(self._field.get_all_components()),
                self.get_sample_times().shape[0])

    def get_sensor_names(self) -> list[str]:
        return self._sensor_names

    #---------------------------------------------------------------------------
    # Truth values - from simulation
    def get_truth_values(self) -> np.ndarray:
        return self._field.sample_field(self._positions,
                                        self._sample_times)


    #---------------------------------------------------------------------------
    # Systematic error calculation functions
    # Only calculated once when set
    def set_sys_err_integrator(self,
                               err_int: SysErrIntegrator) -> None:

        self._sys_err_int = err_int


    def get_systematic_errs(self) -> np.ndarray | None:

        if self._sys_err_int is None:
            return None

        return self._sys_err_int.get_sys_errs_tot()

    #---------------------------------------------------------------------------
    # Random error calculation functions
    def set_rand_err_integrator(self,
                                err_int: RandErrIntegrator) -> None:

        self._rand_err_int = err_int


    def get_random_errs(self) -> np.ndarray | None:

        if self._rand_err_int is None:
            return None

        return self._rand_err_int.get_rand_errs_tot()


    #---------------------------------------------------------------------------
    # Measurement calculations
    def get_measurements(self) -> np.ndarray:

        measurements = self.get_truth_values()
        sys_errs = self.get_systematic_errs()
        rand_errs = self.get_random_errs()

        if sys_errs is not None:
            measurements = measurements + sys_errs

        if rand_errs is not None:
            measurements = measurements + rand_errs

        return measurements


    def get_measurement_data(self) -> MeasurementData:
        measurement_data = MeasurementData()
        measurement_data.measurements = self.get_measurements()
        measurement_data.systematic_errs = self.get_systematic_errs()
        measurement_data.random_errs = self.get_random_errs()
        measurement_data.truth_values = self.get_truth_values()
        return measurement_data


    #---------------------------------------------------------------------------
    # Plotting tools
    def get_visualiser(self) -> pv.PolyData:
        pv_data = pv.PolyData(self._positions)
        pv_data['labels'] = self._sensor_names
        return pv_data

    def plot_time_traces(self,
                         plot_truth: bool = False,
                         plot_sim: bool = False) -> tuple[Any,Any]:
        pp = PlotProps()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        comp = self._field.get_all_components()[0]

        fig, ax = plt.subplots(figsize=pp.single_fig_size,layout='constrained')
        fig.set_dpi(pp.resolution)

        sim_time = self._field.get_time_steps()
        if plot_sim:
            sim_vals = self._field.sample_field(self._positions)
            for ii in range(self.get_num_sensors()):
                ax.plot(sim_time,sim_vals[ii,0,:],'-o',
                    lw=pp.lw/2,ms=pp.ms/2,color=colors[ii])

        if self.get_sample_times() is None:
            samp_time = self._field.get_time_steps()
        else:
            samp_time = self.get_sample_times()

        if plot_truth:
            truth = self.get_truth_values()
            for ii in range(self.get_num_sensors()):
                ax.plot(samp_time,truth[ii,0,:],'-',
                    lw=pp.lw/2,ms=pp.ms/2,color=colors[ii])

        measurements = self.get_measurements()
        for ii in range(self.get_num_sensors()):
            ax.plot(samp_time,measurements[ii,0,:],
                ':+',label=self._sensor_names[ii],
                lw=pp.lw/2,ms=pp.ms/2,color=colors[ii])

        ax.set_xlabel(r'Time, $t$ [s]',
                    fontsize=pp.font_ax_size, fontname=pp.font_name)
        ax.set_ylabel(r'Temperature, $T$ [$\degree C$]',
                    fontsize=pp.font_ax_size, fontname=pp.font_name)

        ax.set_xlim([np.min(samp_time),np.max(samp_time)]) # type: ignore

        plt.grid(True)
        ax.legend()
        ax.legend(prop={"size":pp.font_leg_size},loc='upper left')
        plt.draw()

        return (fig,ax)
