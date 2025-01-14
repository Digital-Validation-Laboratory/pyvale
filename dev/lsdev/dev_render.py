from multiprocessing import cpu_count
from enum import Enum
from dataclasses import dataclass
import bpy
from dev_blendercamera import CameraData

class RenderEngine(Enum):
    """Different render engines on Blender
    """
    CYCLES = "CYCLES"
    EEVEE = "EEVEE"
    WORKBENCH = "WORKBENCH"

@dataclass
class RenderData:
    samples: int | None = None
    engine: RenderEngine = RenderEngine.CYCLES



class Render:
    def __init__(self,
                 RenderData,
                 image_path: str,
                 output_path: str,
                 cam_data: CameraData):
        self.render_data = RenderData
        self.image_path = image_path
        self.output_path = output_path
        self.cam_data = cam_data
        self.scene = bpy.data.scenes['Scene']

    def render_parameters(self,
                          file_name: str,
                          cores: int):
        bpy.context.scene.render.engine = self.render_data.engine.CYCLES.value
        bpy.context.scene.view_settings.look = 'AgX - Greyscale'
        bpy.context.scene.cycles.samples = self.render_data.samples
        self.scene.render.resolution_x = self.cam_data.sensor_px[0]
        self.scene.render.resolution_y = self.cam_data.sensor_px[1]
        self.scene.render.filepath =  str(self.image_path / file_name)
        self.scene.render.threads_mode = 'FIXED'
        self.scene.render.threads = cores
        self.scene.cycles.use_denoising = False # To make rendering faster

        bpy.context.scene.render.image_settings.file_format = 'TIFF'

        bpy.ops.render.render(write_still=True)

    def render_image(self, name: int, image_count: int, part):

        file_name = name + '_' + str(image_count) + '.tiff'
        cores = int(cpu_count())
        self.render_parameters(file_name, cores)
        self._write_progress(image_count, part)

    def _write_progress(self, image_count: int, part):
        if image_count == 0:
            report = open(self.output_path, 'w', encoding='utf-8')
        else:
            report = open(self.output_path, 'a', encoding='utf-8')
        report.write('\nOn render: ' + str(image_count))
        report.write('\nPart dimensions: ' + str(part.dimensions))
        report.write('\nPart location: ' + str(part.location))
        report.write('\nPart rotation: ' + str(part.rotation_euler))
        report.write('\n')
        report.close()



