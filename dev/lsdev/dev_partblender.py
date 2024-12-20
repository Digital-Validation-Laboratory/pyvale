import numpy as np
import bpy
from mooseherder.simdata import SimData

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
        if sim_data is not None:
            self._initialise_nodes_elements(elements, nodes)




    def _initialise_nodes_elements(self, elements, nodes):
        if elements is None:
            self.elements = self._get_elements()
        else:
            self.elements = elements

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

        zero_index_elements = elements - 1 # Blender has a zero base index

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

        zero_index_nodes = nodes  # Blender has a zero base index

        centred = self._centre_nodes(zero_index_nodes)

        return centred



    def simdata_to_part(self):
        """Creates an object from the mesh information in the SimData object
        """

        mesh = bpy.data.meshes.new("part")
        mesh.from_pydata(self.nodes, [], self.elements, shade_flat=True)
        mesh.validate(verbose=True, clean_customdata=True)
        part = bpy.data.objects.new("specimen", mesh)
        bpy.context.scene.collection.objects.link(part)

        return part

    def import_from_stl(self):
        bpy.ops.wm.stl_import(filepath=self.filename)

        part = bpy.context.selected_objects[0]
        return part

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