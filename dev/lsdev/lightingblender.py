from dataclasses import dataclass
from enum import Enum
import numpy as np
import bpy

class LightType(Enum):
    POINT = 'POINT'
    SUN = 'SUN'
    SPOT = 'SPOT'
    AREA = 'AREA'

@dataclass
class LightData():
    type: LightType | None = None
    position: np.ndarray | None = None
    orientation: np.ndarray | None = None
    energy: int | None = None


class BlenderLight():
    def __init__(self):
        self.light_data = LightData
        self._light_ob = None
        self._light = None

    def _create_light(self):
        self._light = bpy.data.lights.new(name='spot', type='SPOT')
        self._light_ob = bpy.data.objects.new(name='Spot', object_data=light)
        bpy.context.collection.objects.link(self._light_ob)

    def _get_light_object(self):
        self._light_ob = self._light_ob

    def _get_light(self):
        self._light = self.light

    def _set_location(self):
        self._light_ob.location = self.light_data.position

    def _set_rotation(self):
        self._light_ob.rotation_mode = 'EULER'
        self._light_ob.rotation_euler = self.light_data.orientation

    def _set_energy(self):
        self._light.energy = self.light_data.energy

    def add_light(self):
        self.create_light()
        self.set_location()
        self.set_rotation()
        self.set_energy()


