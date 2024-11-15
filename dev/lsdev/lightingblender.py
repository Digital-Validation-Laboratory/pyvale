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
        self.light_ob = None
        self.light = None

    def create_light(self):
        self.light = bpy.data.lights.new(name='spot', type='SPOT')
        self.light_ob = bpy.data.objects.new(name='Spot', object_data=light)
        bpy.context.collection.objects.link(self.light_ob)

    def get_light_object(self):
        self.light_ob = self.light_ob

    def get_light(self):
        self.light = self.light

    def set_location(self):
        self.light_ob.location = self.light_data.position

    def set_rotation(self):
        self.light_ob.rotation_mode = 'EULER'
        self.light_ob.rotation_euler = self.light_data.orientation

    def set_energy(self):
        self.light.energy = self.light_data.energy

    def add_light(self):
        self.create_light()
        self.set_location()
        self.set_rotation()
        self.set_energy()


