import bpy
from camera import CameraData, CameraBlender
from lightingblender import LightData, BlenderLight

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
        light = BlenderLight.add_light()
        return light

    def add_camera(self):
        # TODO: Set variables in dataclass
        camera = CameraBlender.add_camera()
        return camera




