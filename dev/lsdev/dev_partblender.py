import os
import numpy as np
from pathlib import Path
import bpy
from mooseherder.simdata import SimData
import pyvale

class BlenderPart:
    """Creates an object in Blender
    """
    def __init__(self,
                 sim_data: SimData | None = None,
                 elements:np.ndarray | None = None,
                 nodes: np.ndarray | None = None,
                 filename: str | None = None):
        self.sim_data = sim_data
        self.filename = filename
        self.elements = elements
        self.nodes = nodes

    def _initialise_nodes_elements(self, elements, nodes):
        if elements is None:
            self.elements = self._get_elements() * 1000
        else:
            self.elements = elements * 1000

        if nodes is None:
            self.nodes = self._get_nodes() * 1000
        else:
            self.nodes = nodes * 1000


    def _get_elements(self):
        """Gets the connectivity table from the SimData object and converts it
           into a format Blender can read
        """
        connect = self.sim_data.connect[np.str_('connect1')]

        elements = connect.T

        zero_index_elements = elements -1 # Blender has a zero base index

        return zero_index_elements

    def _get_nodes(self):
        """Gets the node coordinates from the SimData object and converts it
           into a format Blender can read
        """
        nodes = self.sim_data.coords

        zero_index_nodes = nodes

        centred = centre_nodes(zero_index_nodes)

        return centred

    def _get_spat_dim(self):
        nodes = self.sim_data.coords
        check_if_2d = np.count_nonzero(nodes, axis=0)
        if check_if_2d[2] == 0:
            spat_dim = 2
        else:
            spat_dim = 3
        return spat_dim

    def _get_components(self) -> tuple:
        node_vars = self.sim_data.node_vars
        node_vars_names = list(node_vars.keys())
        components = []
        if 'disp_x' in node_vars_names:
            components.append('disp_x')
        if 'disp_y' in node_vars_names:
            components.append('disp_y')
        if 'disp_z' in node_vars_names:
            components.append('disp_z')
        # if 'temperature' in node_vars_names:
        #     components.append('temperature')
        components = tuple(components)
        if len(components) == 0:
            components = None

        return components

    def _simdata_to_pvsurf(self, components, spat_dim):
        self.sim_data.coords = self.sim_data.coords * 1000
        (pv_grid, pv_grid_vis) = pyvale.conv_simdata_to_pyvista(self.sim_data,
                                                                components,
                                                                spat_dim=spat_dim)
        pv_surf = pv_grid.extract_surface()
        # tri_surf = pv_surf.triangulate()
        # tri_surf.plot(show_edges=True, line_width=2)

        return pv_surf, pv_grid


    def _pv_surf_to_obj(self, pv_surf):
        save_path = Path().cwd() / "test_output"
        if not save_path.is_dir():
            save_path.mkdir()
        name = "test_mesh.obj" #Filetype changed
        save_file = save_path / name

        all_files = os.listdir(save_path)
        for ff in all_files:
            if name == ff:
                os.remove(save_path / ff)

        self.filename = str(save_file)
        pv_surf.save(save_file, binary=False)


    def import_from_obj(self, pv_surf = None):
        if self.filename is None:
            self._pv_surf_to_stl(pv_surf)

        bpy.ops.wm.obj_import(filepath=self.filename) #Changed filetype

        part = bpy.context.selected_objects[0]
        return part

def centre_nodes(nodes):
        max = np.max(nodes, axis=0)
        min = np.min(nodes, axis=0)
        middle = max - ((max - min) / 2)
        centred = np.subtract(nodes, middle)
        return centred
