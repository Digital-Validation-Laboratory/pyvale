"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
import numpy as np
from pyvale.core.field import IField


class ISensorArray(ABC):
    """Interface (abstract base class) for an array of sensors of the same
    type sampling a given physical field.
    """

    @abstractmethod
    def get_measurement_shape(self) -> tuple[int,int,int]:
        """Abstract method. Gets the shape of the measurement array:
        shape=(num_sensors,num_field_components,num_time_steps).

        The number of sensors is specified by the user with a SensorData object.
        The number of field components is dependent on the field being sampled
        (i.e. 1 for a scalar field and 3 for a vector field in 3D). The number
        of time steps is specified by the user in the SensorData object or
        defaults to the time steps taken from the simulation.

        Returns
        -------
        tuple[int,int,int]
            Shape of the measurement array as (num_sensors,
            num_field_components,num_time_steps)
        """
        pass

    @abstractmethod
    def get_field(self) -> IField:
        """Abstract method. Gets the field object that this array of sensors is
        sampling to simulate measurements.

        Returns
        -------
        IField
            A field object interface.
        """
        pass

    @abstractmethod
    def calc_measurements(self) -> np.ndarray:
        """Abstract method. Calculates measurements as: measurement = truth +
        systematic errors + random errors. The truth is calculated once and is
        interpolated from the input simulation field. The errors are calculated
        based on the user specified error chain.

        NOTE: this method will sample all probability distributions in the error
        chain returning a new simulated experiment for this sensor array.

        Returns
        -------
        np.ndarray
            The calculated measurements for this sensor array with shape:
            (num_sensors,num_field_components,num_time_steps)
        """
        pass

    @abstractmethod
    def get_measurements(self) -> np.ndarray:
        """Abstract method. Returns the current set of simulated measurements if
        theses have been calculated. If these have not been calculated then
        'calc_measurements()' is called and a set of measurements in then
        returnes.

        NOTE: this method does not sample from probability distributions in the
        error chain and directly returns the current set of measurements if they
        exist.

        Returns
        -------
        np.ndarray
            The calculated measurements for this sensor array with shape:
            (num_sensors,num_field_components,num_time_steps)
        """
        pass

