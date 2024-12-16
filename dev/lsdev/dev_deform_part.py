import numpy as np
from pathlib import Path
import bpy
import mooseherder as mh
from mooseherder import SimData
from dev_partblender import BlenderPart

class DeformMesh:
    def __init__(self, sim_data:SimData, defgrad):
        self.sim_data = sim_data
        self.defgrad = defgrad
        self.nodes = self._get_nodes()

    def _get_nodes(self):
        mesh_builder = BlenderPart(self.sim_data)
        nodes = mesh_builder._get_nodes()
        return nodes

    def map_coords(self):
        mid = np.max(self.nodes, axis=0)  / 2
        centred = np.subtract(self.nodes, mid)

        defgrad_inv = np.linalg.inv(self.defgrad)

        coords_new = np.einsum('ij,nj->ni', defgrad_inv, centred)

        return coords_new

class DeformPart:
    def __init__(self, part, deformed_nodes):
        self.part = part
        self.deformed_nodes = deformed_nodes * 1000

    def set_new_frame(self):
        frame_incr = 20
        ob = bpy.context.view_layer.objects.active
        if ob is None:
            bpy.context.objects.active = self.part

        current_frame = bpy.context.scene.frame_current
        current_frame += frame_incr
        bpy.context.scene.frame_set(current_frame)

        bpy.data.shape_keys['Key'].eval_time = current_frame
        self.part.data.shape_keys.keyframe_insert('eval_time', frame=current_frame)
        bpy.context.scene.frame_end = current_frame

    def deform_part(self):
        if self.part.data.shape_keys is None:
            self.part.shape_key_add()
            self.set_new_frame()
        sk = self.part.shape_key_add()
        self.part.data.shape_keys.use_relative = False

        n_nodes_layer = int(len(self.part.data.vertices))
        all_nodes = np.array([sk.data[i].co for i in range(len(self.part.data.vertices))])
        first_layer = all_nodes[0:n_nodes_layer, :]

        count = 0
        for i in range(len(self.part.data.vertices)):
            if i < n_nodes_layer:
                sk.data[i].co = self.deformed_nodes[i]
                count += 1
            else:
                dist = np.linalg.norm(first_layer - sk.data[i].co, axis=1)
                cn = np.argmin(dist)
                sk.data[i].co = self.deformed_nodes[cn]

        return self.part







