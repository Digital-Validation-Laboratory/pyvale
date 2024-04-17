'''
================================================================================
pycave: the python computer aided validation engine.
License: LGPL-2.1
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Callable, Any
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from pycave.field import Field
from pycave.sensorarray import SensorArray, MeasurementData
from pycave.plotprops import PlotProps


class ThermocoupleArray(SensorArray):
    def __init__(self,
                 positions: np.ndarray,
                 field: Field,
                 sample_times: np.ndarray | None = None
                 ) -> None:

        self._positions = positions
        self._field = field
        self._sample_times = sample_times

        self._sys_err_func = None
        self._sys_errs = None

        self._rand_err_func = None

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


    def get_measurement_shape(self) -> tuple[int,int]:
        return (self.get_num_sensors(),
                self.get_sample_times().shape[0])

    def get_sensor_names(self) -> list[str]:
        return self._sensor_names


    #---------------------------------------------------------------------------
    # Truth values - from simulation
    def get_truth_values(self) -> np.ndarray:
        return self._field.sample(self._positions,self._sample_times)


    #---------------------------------------------------------------------------
    # Systematic error calculation functions
    def set_uniform_systematic_err_func(self, low: float, high: float) -> None:

        def sys_err_func(size: tuple) -> np.ndarray:
            sys_errs = np.random.default_rng().uniform(low=low,
                                                    high=high,
                                                    size=(size[0],1))
            sys_errs = np.tile(sys_errs,(1,size[1]))
            return sys_errs

        self._sys_err_func = sys_err_func
        self._sys_errs = self._sys_err_func(size=self.get_measurement_shape())


    def set_custom_systematic_err_func(self, sys_fun: Callable | None = None
                                ) -> np.ndarray | None:

        self._sys_err_func = sys_fun

        if self._sys_err_func is None:
            self._sys_errs = None
            return None

        self._sys_errs = self._sys_err_func(size=self.get_measurement_shape())
        return self._sys_errs


    def get_systematic_errs(self) -> np.ndarray | None:

        if self._sys_err_func is None:
            return None

        return self._sys_errs

    #---------------------------------------------------------------------------
    # Random error calculation functions
    def set_normal_random_err_func(self, std_dev: float) -> None:

        self._rand_err_func = partial(np.random.default_rng().normal,
                                        loc=0.0,
                                        scale=std_dev)


    def set_custom_random_err_func(self, rand_fun: Callable | None = None
                                   ) -> None:
        self._rand_err_func = rand_fun


    def get_random_errs(self) -> np.ndarray | None:
        if self._rand_err_func is None:
            return None

        return self._rand_err_func(size=self.get_measurement_shape())


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

        fig, ax = plt.subplots(figsize=pp.single_fig_size,layout='constrained')
        fig.set_dpi(pp.resolution)

        sim_time = self._field.get_time_steps()
        if plot_sim:
            sim_vals = self._field.sample(self._positions)
            for ii in range(self.get_num_sensors()):
                ax.plot(sim_time,sim_vals[ii,:],'-o',
                    lw=pp.lw/2,ms=pp.ms/2,color=colors[ii])

        if self.get_sample_times() is None:
            samp_time = self._field.get_time_steps()
        else:
            samp_time = self.get_sample_times()

        if plot_truth:
            truth = self.get_truth_values()
            for ii in range(self.get_num_sensors()):
                ax.plot(samp_time,truth[ii,:],'-',
                    lw=pp.lw/2,ms=pp.ms/2,color=colors[ii])

        measurements = self.get_measurements()
        for ii in range(self.get_num_sensors()):
            ax.plot(samp_time,measurements[ii,:],
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
