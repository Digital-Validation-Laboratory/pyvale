from dataclasses import dataclass
import bpy

@dataclass
class MaterialData():
    roughness: float | None = 0.5
    metallic: float | None = 0
    # TODO: add other material properties to here

class BlenderMaterial():
    def __init__(self, MaterialData, object, image_path):
        self.mat_data = MaterialData
        self.object = object
        self.image_path = image_path

    def _clear_nodes(self):
        self.object.select_set(True)
        mat = bpy.data.materials.new(name='Material') # add this to init?
        mat.use_nodes = True
        tree = mat.node_tree
        nodes = tree.nodes
        nodes.clear()

    def _set_basic_material(self):
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        bsdf.inputs['Roughness'].default_value = self.mat_data.roughness
        bsdf.inputs['Metallic'].default_value = self.mat_data.metallic

    def _set_image_texture(self):
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (0, 0)




