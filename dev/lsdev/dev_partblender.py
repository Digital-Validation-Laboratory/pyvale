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

    def _centre_nodes(self, nodes):
        max = np.max(nodes, axis=0)
        min = np.min(nodes, axis=0)
        middle = max - ((max - min) / 2)
        centred = np.subtract(nodes, middle)
        return centred

    def _get_nodes(self):
        """Gets the node coordinates from the SimData object and converts it
           into a format Blender can read
        """
        nodes = self.sim_data.coords

        zero_index_nodes = nodes

        centred = self._centre_nodes(zero_index_nodes)

        return centred



    def simdata_to_part(self):
        """Creates an object from the mesh information in the SimData object
        """
        nodes = self._get_nodes() * 1000
        elements = self._get_elements()
        mesh = bpy.data.meshes.new("part")
        mesh.from_pydata(nodes, [], elements, shade_flat=True)
        mesh.validate(verbose=True, clean_customdata=True)
        part = bpy.data.objects.new("specimen", mesh)
        bpy.context.scene.collection.objects.link(part)

        return part

    def _simdata_to_stl(self):
        self.sim_data.coords = self.sim_data.coords * 1000
        (pv_grid, pv_grid_vis) = pyvale.conv_simdata_to_pyvista(self.sim_data,
                                                                None,
                                                                spat_dim=3)

        pv_surf = pv_grid.extract_surface()
        surface_points = pv_surf.points
        centre_points = self._centre_nodes(surface_points)

        save_path = Path().cwd() / "test_output"
        if not save_path.is_dir():
            save_path.mkdir()
        name = "test_mesh.stl"
        save_file = save_path / name

        all_files = os.listdir(save_path)
        for ff in all_files:
            if name == ff:
                os.remove(save_path / ff)

        self.filename = str(save_file)
        pv_surf.save(save_file, binary=False)


        return centre_points


    def import_from_stl(self):
        if self.filename is None:
            points = self._simdata_to_stl()

        bpy.ops.wm.stl_import(filepath=self.filename)

        part = bpy.context.selected_objects[0]
        return part, points

    def add_thickness(self, part): # Not sure if this is necessary
        part["solidify"] = True
        part["thickness"] = 1

        if part["solidify"]:
            ob = bpy.context.view_layer.objects.active
            if ob is None:
                bpy.context.view_layer.objects.active = part
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.solidify(thickness=1)
            bpy.ops.object.editmode_toggle()

        return part