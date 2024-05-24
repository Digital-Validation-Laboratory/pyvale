'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import warnings
import numpy as np

class CameraData:
    def __init__(self,num_px: np.ndarray | None = None,
                 bits: int = 8,
                 m_per_px: float = 1.0e-3) -> None:
        # Core params
        if num_px is None:
            self._num_px = np.array((1000,1000))
        else:
            self._num_px = num_px

        self._bits = bits
        self._m_per_px = m_per_px

        # Calculated parameters
        self._fov = self._m_per_px*self._num_px
        self._dyn_range = 2**self._bits
        self._background = self._dyn_range/2

        # Region of Interest (ROI)
        self._roi_cent = (True,True)
        self._roi_len = self._fov
        self._roi_loc = np.array([0.0,0.0])

    @property
    def num_px(self):
        return self._num_px

    @property
    def bits(self):
        return self._bits
    @property
    def m_per_px(self):
        return self._m_per_px

    @property
    def fov(self):
        return self._fov

    @property
    def dyn_range(self):
        return self._dyn_range

    @property
    def background(self):
        return self._background

    @property
    def roi_len(self):
        return self._roi_len

    @property
    def roi_loc(self):
        return self._roi_loc

    @property
    def roi_cent(self):
        return self._roi_cent

    @num_px.setter
    def num_px(self,in_px):
        self._num_px = in_px
        self._fov = self._m_per_px*self._num_px

    @bits.setter
    def bits(self,in_bits):
        self._bits = in_bits
        self._dyn_range = 2**self._bits
        self._background = self._dyn_range*0.5

    @background.setter
    def background(self,background):
        self._background = background

    @m_per_px.setter
    def m_per_px(self,in_calib):
        self._m_per_px = in_calib
        self._fov = self._m_per_px*self._num_px

    @roi_len.setter
    def roi_len(self,in_len):
        self._roi_len = in_len
        self._cent_roi()

    @roi_loc.setter
    def roi_loc(self,in_loc):
        if sum(self._roi_cent) > 0:
            warnings.warn('ROI is centered, cannot set ROI location. Update centering flags to adjust.')
        else:
            self._roi_loc = in_loc

    @roi_cent.setter
    def roi_cent(self,in_flags):
        self._roi_cent = in_flags
        self._cent_roi()

    def _cent_roi(self):
        if self._roi_cent[0] == True:
            self._roi_loc[0] = (self._fov[0] - self._roi_len[0])/2
        if self._roi_cent[1] == True:
            self._roi_loc[1] = (self._fov[1] - self._roi_len[1])/2
