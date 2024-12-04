from multiprocessing import cpu_count
from enum import Enum
from dataclasses import dataclass
from camera import CameraData
import bpy

class RenderEngine(Enum):
    """Different render engines on Blender
    """
    # TODO: Check if these names are actually correct
    CYCLES = "CYCLES"
    EEVEE = "EEVEE"
    WORKBENCH = "WORKBENCH"

@dataclass
class RenderData:
    samples: int | None = None
    engine: RenderEngine = RenderEngine.CYCLES



class Render:
    def __init__(self, RenderData, image_path, output_path):
        self.render_data = RenderData
        self.image_path = image_path
        self.output_path = output_path
        self.scene = bpy.data.scenes['Scene'] # Work out how to integrate this to scene creation

    def render_parameters(self,
                          file_name,
                          cores):
        bpy.context.scene.render.engine = self.render_data.engine.CYCLES.value
        bpy.context.scene.cycles.samples = self.render_data.samples
        self.scene.render.resolution_x = CameraData.sensor_px[0]
        self.scene.render.resolution_y = CameraData.sensor_px[1]
        self.scene.render.filepath =  str(self.image_path / file_name)
        self.scene.render.threads_mode = 'FIXED'
        self.scene.render.threads = cores

        bpy.context.scene.render.image_settings.file_format = 'TIFF'

        bpy.ops.render.render(write_still=True)

    def render_image(self, name, image_count):

        file_name = name + '_' + str(image_count) + '.tiff'
        cores = int(cpu_count())
        self.render_parameters(file_name, cores)


