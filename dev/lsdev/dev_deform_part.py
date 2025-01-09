import numpy as np
from pathlib import Path
import bpy
import pyvale
import mooseherder as mh
from mooseherder import SimData
from dev_partblender import BlenderPart

class DeformMesh:
    def __init__(self,sim_data: SimData):
        self.sim_data = sim_data

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

    def add_displacement(self, timestep: int, nodes: np.ndarray):
        node_var_names = self._get_node_vars()
        disps = self._check_for_displacements(node_var_names)
        if True in disps.values():
            shape = self.sim_data.coords.shape
            added_disp = np.empty(shape)
            dim = 0
            for disp, value in disps.items():
                if value is True:
                    added_disp_1d = self.sim_data.node_vars[disp][:, timestep]
                    added_disp[:, dim] = added_disp_1d
                    print(f"{added_disp=}")
            added_disp_suface = self._nodes_to_surface_mesh(added_disp)
            print(f"{added_disp_suface=}")

            deformed_nodes = nodes + added_disp_suface
            return deformed_nodes
        else:
            return None

    def _nodes_to_surface_mesh(self, deformed_nodes):
        self.sim_data.coords = deformed_nodes
        (pv_grid, pv_grid_vis) = pyvale.conv_simdata_to_pyvista(self.sim_data,
                                                                None,
                                                                spat_dim=3)
        pv_surf = pv_grid.extract_surface()
        surface_points = pv_surf.points

        return surface_points

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
        disp = {'disp_x': False, 'disp_y': False, 'disp_z': False}

        if 'disp_x' in node_var_names:
            disp['disp_x'] = True
        if 'disp_y' in node_var_names:
            disp['disp_y'] = True
        if 'disp_z' in node_var_names:
            disp['disp_z'] = True

        return disp

    def add_displacement(self, timestep: int):
        nodes = self._get_nodes()
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
            check_if_2d = np.count_nonzero(nodes, axis=0)
            if check_if_2d[2] != 0:
                deformed_surface = self._nodes_to_surface_mesh(deformed_nodes)
            else:
                deformed_surface = deformed_nodes
            return deformed_surface
        else:
            return None

    def _nodes_to_surface_mesh(self, deformed_nodes):
        self.sim_data.coords = deformed_nodes
        (pv_grid, pv_grid_vis) = pyvale.conv_simdata_to_pyvista(self.sim_data,
                                                                None,
                                                                spat_dim=3)
        pv_surf = pv_grid.extract_surface()
        surface_points = pv_surf.points

        return surface_points




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







