import bpy
import numpy as np
from mooseherder import SimData
from dev_blendercamera import CameraData, CameraBlender
from dev_lightingblender import LightData, BlenderLight
from dev_partblender import BlenderPart
from dev_objectmaterial import MaterialData, BlenderMaterial

class BlenderScene:
    def __init__(self):
        bpy.ops.wm.read_factory_settings(use_empty=True) #Blender starts with empty scene, without default objects
        self.default_settings()

    def default_settings(self):
        """Sets the default settings in Blender:
        - Sets the units to mm

        # part = partmaker.add_thickness(part=part)
        - Sets the background to black
        """
        bpy.context.scene.unit_settings.scale_length = 0.001
        bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

        new_world = bpy.data.worlds.new('World')
        bpy.context.scene.world = new_world
        new_world.use_nodes = True
        node_tree = new_world.node_tree
        nodes = node_tree.nodes

        nodes.clear()
        bg_node = nodes.new(type='ShaderNodeBackground')
        bg_node.inputs[0].default_value = [0.5, 0.5, 0.5, 1]
        bg_node.inputs[1].default_value = 0

    def add_light(self, light_data: LightData):
        lightmaker = BlenderLight(light_data)
        light = lightmaker.add_light()
        return light

    def add_camera(self, cam_data: CameraData):
        cameramaker = CameraBlender(cam_data)
        camera = cameramaker.add_camera()

        return camera

    def add_part(self,
                 sim_data: SimData,
                 elements: np.ndarray | None = None,
                 nodes: np.ndarray | None = None,
                 filename: str | None = None):
        check_if_2d = np.count_nonzero(sim_data.coords, axis=0)
        if check_if_2d[2] == 0:
            part = self.add_simdata_part(sim_data, elements, nodes)
        else:
            part, points = self.add_stl_part(sim_data=sim_data, filename=filename)

        return part

    def add_simdata_part(self,
                 sim_data: SimData,
                 elements:np.ndarray | None = None,
                 nodes: np.ndarray | None = None):
        partmaker = BlenderPart(sim_data, elements, nodes)
        part = partmaker.simdata_to_part()

        set_origin(part)

        return part

    def add_stl_part(self, filename:str | None = None, sim_data: SimData | None = None):
        partmaker = BlenderPart(filename=filename, sim_data=sim_data)
        part, points = partmaker.import_from_stl()
        set_origin(part)
        return part, points


    def set_part_location(self, part, location: tuple):
        print(f"{part=}")
        z_location = int(part.dimensions[2])
        part.location = (location[0], location[1], (location[2] - z_location))

    def set_part_roation(self, part, rotation: tuple):
        part.rotation_mode = 'XYZ'
        part.rotation_euler = rotation

    def add_material(self, mat_data: MaterialData, part, image_path: str):
        materialmaker = BlenderMaterial(mat_data, part, image_path)
        mat = materialmaker.add_material()

        return mat


    def save_model(self, filepath: str):
        '''
        Method to save the blender model to a file
        '''
        if filepath is not None:
            bpy.ops.wm.save_as_mainfile(filepath=filepath)

def set_origin(part):
        bpy.ops.object.select_all(action='DESELECT')
        part.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
        # dimensions = part.dimensions
        # z_location = int(dimensions[2])
        # if z_location != 0:
        #     location = (0, 0, (0 - z_location))
        #     part.location(location)





