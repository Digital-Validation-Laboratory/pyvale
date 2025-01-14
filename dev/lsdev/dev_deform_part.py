import numpy as np
from pathlib import Path
import bpy
import pyvale
import mooseherder as mh
from mooseherder import SimData
from dev_partblender import BlenderPart, centre_nodes

class DeformMesh:
    def __init__(self,pv_surf, spat_dim, components):
        self.pv_surf = pv_surf
        self.spat_dim = spat_dim
        self.components = components

    def add_displacement(self, timestep: int, nodes):
        if set(self.components).issubset(self.pv_surf.array_names):
            added_disp = np.zeros_like(nodes)
            dim = 0
            for component in self.components:
                added_disp_1d = self.pv_surf.get_array(component)[:, timestep]
                added_disp[:, dim] = added_disp_1d
                dim += 1
            deformed_nodes = nodes + added_disp
            deformed_nodes = centre_nodes(deformed_nodes)
            added_disp = centre_nodes(added_disp)
            return deformed_nodes, added_disp
        else:
            return None

class DeformSimData:
    """Class to test old way of deforming mesh - only for 2D mesh
    """    
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
        disp = {'disp_x': False, 'disp_y': False, 'disp_z': False}

        if 'disp_x' in node_var_names:
            disp['disp_x'] = True
        if 'disp_y' in node_var_names:
            disp['disp_y'] = True
        if 'disp_z' in node_var_names:
            disp['disp_z'] = True

        return disp

    def add_displacement(self, timestep: int, nodes):
        nodes = nodes
        node_var_names = self._get_node_vars()
        disps = self._check_for_displacements(node_var_names)
        if True in disps.values():
            deformed_nodes = nodes
            dim = 0
            for disp, value in disps.items():
                if value is True:
                    added_disp = self.sim_data.node_vars[disp][:, timestep]
                    node_dim = nodes[:, dim]
                    deformed_nodes[:, dim] = node_dim + added_disp
                    dim += 1
            return deformed_nodes
        else:
            return None



class DeformPart:
    def __init__(self, part, deformed_nodes):
        self.part = part
        self.deformed_nodes = deformed_nodes

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
        for i in range(len(self.part.data.vertices)):
            if i < n_nodes_layer:
                sk.data[i].co = self.deformed_nodes[i]
        return self.part







