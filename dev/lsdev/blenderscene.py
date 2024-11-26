import bpy
from camera import CameraData, CameraBlender
from lightingblender import LightData, BlenderLight
from dev_partblender import BlenderPart

class BlenderScene:
    def __init__(self):
        self.objects = list()
        self.lights = list()
        self.cameras = list()
        bpy.ops.wm.read_factory_settings(use_empty=True) #Blender starts with empty scene, without default objects
        self.default_settings()

    def default_settings(self):
        """Sets the default settings in Blender:
        - Sets the units to mm
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
        bg_node = nodes.new(type='ShaderNodeBackground')  # Add Background node
        bg_node.inputs[0].default_value = [0.5, 0.5, 0.5, 1]
        bg_node.inputs[1].default_value = 0

    def add_light(self):
        # TODO: Set variables in dataclass
        lightmaker = BlenderLight()
        light = lightmaker.add_light()
        return light

    def add_camera(self, location=None, rotation=None):
        # TODO: Set variables in dataclass
        cameramaker = CameraBlender()
        camera = cameramaker.add_camera()

        return camera

    def add_part(self, sim_data):
        # Structure of this method is confused
        partmaker = BlenderPart(sim_data)
        part = partmaker.simdata_to_part()

        part.select_set(True)

        part.select_set(False)

        self._set_origin(part)

        return part

    def set_part_location(self, part, location):
        part.location = location

    def set_part_roation(self, part, rotation):
        part.rotation_mode = 'XYZ'
        part.rotation_euler = rotation

    def _set_origin(self, part):
        # Not sure if this is necessary
        bpy.ops.object.select_all(action='DESELECT')
        part.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

    def save_model(self, filepath):
        '''
        Method to save the blender model to a file
        '''
        if filepath is not None:
            bpy.ops.wm.save_as_mainfile(filepath=filepath)





