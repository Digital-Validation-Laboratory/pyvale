import numpy as np
import bpy
from mooseherder.simdata import SimData

class BlenderPart:
    def __init__(self, sim_data: SimData):
        self.sim_data = sim_data

    def _get_elements(self):
        connect = self.sim_data.connect[np.str_('connect1')] # Write this so not hardcoded
        elements = connect.T

        return elements

    def _get_nodes(self):
        nodes = self.sim_data.coords

        return nodes




    def simdata_to_part(self):
        elements = self._get_elements()
        nodes = self._get_nodes()

        mesh = bpy.data.meshes.new("part")
        mesh.from_pydata(nodes, [], elements)
        part = bpy.data.objects.new("specimen", mesh)
        bpy.context.scene.collection.objects.link(part)

        return part

    def set_part_location(part, position):
        part.location = position

    def set_part_rotation(part, rotation):
        part.rotation_mode = "EULER"
        part.rotation_euler = rotation
