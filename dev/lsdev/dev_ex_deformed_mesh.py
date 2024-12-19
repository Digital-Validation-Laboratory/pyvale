import os
from pathlib import Path
import numpy as np
import mooseherder as mh
from dev_blenderscene import BlenderScene, set_origin
from dev_partblender import *
from dev_blendercamera import CameraData
from dev_lightingblender import LightData, LightType
from dev_objectmaterial import MaterialData
from dev_render import RenderData, Render
from dev_deform_part import DeformMesh, DeformSimData, DeformPart

def main() -> None:
    # Making Blender scene
    data_path = Path('src/pyvale/simcases/case22_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    dir = Path.cwd() / 'dev/lsdev/blender_files'
    filename = 'case22_deformed.blend'
    filepath = dir / filename
    all_files = os.listdir(dir)
    for ff in all_files:
        if filename == ff:
            os.remove(dir / ff)

    filepath = str(filepath)

    scene = BlenderScene()

    part_location = (0, 0, 0)
    part = scene.add_part(sim_data)
    scene.set_part_location(part, part_location)

    mat_data = MaterialData()
    image_path = '/home/lorna/speckle_generator/images/blender_image_texture.tiff'
    mat = scene.add_material(mat_data, part, image_path)

    sensor_px = (2452, 2056)
    cam_position = (0, 0, 200)
    focal_length = 15.0
    cam_data = CameraData(sensor_px=sensor_px,
                          position=cam_position,
                          focal_length=focal_length)

    camera = scene.add_camera(cam_data)

    type = LightType.POINT
    light_position = (0, 0, 200)
    energy = 400 * (10)**3
    light_data = LightData(type=type,
                           position=light_position,
                           energy=energy)

    light = scene.add_light(light_data)

    #---------------------------------------------------------------------------
    # Rendering images
    image_path = Path.cwd() / 'dev/lsdev/rendered_images/Deform_from_moose'
    output_path = Path.cwd() / 'dev/lsdev/rendered_images'


    render_data = RenderData(samples=1)
    render = Render(render_data, image_path=image_path, output_path=output_path, cam_data=cam_data)

    render_counter = 0
    render_name = 'ref_image'

    render.render_image(render_name, render_counter)

    #---------------------------------------------------------------------------
    # Deform mesh
    timesteps = sim_data.time.shape[0]
    for timestep in range(timesteps):
        timestep += 1 # Adding at start of loop as timestep = 0 is the original mesh
        meshdeformer = DeformSimData(sim_data)
        deformed_nodes = meshdeformer.add_displacement(timestep)

        if deformed_nodes is not None:
            partdeformer = DeformPart(part, deformed_nodes)
            part = partdeformer.deform_part()
            partdeformer.set_new_frame()

            render_name = 'def_sim_data'
            render.render_image(render_name, timestep)


    # scene.save_model(filepath)



if __name__ == "__main__":
    main()