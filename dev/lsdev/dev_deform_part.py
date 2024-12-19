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

class DeformSimData:
    def __init__(self, sim_data: SimData):
        self.sim_data = sim_data

    def _get_nodes(self):
        mesh_builder = BlenderPart(self.sim_data)
        nodes = mesh_builder._get_nodes()
        return nodes

    def _get_node_vars(self):
        node_vars = self.sim_data.node_vars
        node_vars_names = list(node_vars.keys())
        return node_vars_names

    def _check_for_displacements(self, node_var_names: list):
        disp = {'x_disp': False, 'y_disp': False, 'z_disp': False}

        if 'disp_x' in node_var_names:
            disp['x_disp'] = True
        if 'disp_y' in node_var_names:
            disp['y_disp'] = True
        if 'disp_z' in node_var_names:
            disp['z_disp'] = True

        return disp

    def add_displacement(self, timestep: int):
        nodes = self._get_nodes()
        node_var_names = self._get_node_vars()
        disp = self._check_for_displacements(node_var_names)
        if True in disp.values():
            deformed_nodes = nodes
            dim = 0
            for disp in disp:
                if disp is True:
                    added_disp = self.sim_data.node_vars[disp][timestep]
                    node_dim = nodes[dim]
                    deformed_nodes[dim] = node_dim + added_disp
                    dim += 1
            print(f"{deformed_nodes=}")
            return deformed_nodes
        else:
            return None





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







