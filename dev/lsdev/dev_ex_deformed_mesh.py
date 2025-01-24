import os
from pathlib import Path
import numpy as np
import time
import mooseherder as mh
from dev_partblender import BlenderPart
from dev_blenderscene import BlenderScene, set_origin
from dev_partblender import *
from dev_blendercamera import CameraData
from dev_lightingblender import LightData, LightType
from dev_objectmaterial import MaterialData
from dev_render import RenderData, Render, RenderEngine
from dev_deform_part import DeformMesh, DeformPart

def main() -> None:
    # Making Blender scene
    data_path = Path('src/pyvale/data/case18_1_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()

    dir = Path.cwd() / 'dev/lsdev/blender_files'
    name = 'case_18_deformed' # Give this better name
    filename = name + '.blend'
    filepath = dir / filename
    all_files = os.listdir(dir)
    for ff in all_files:
        if filename == ff:
            os.remove(dir / ff)

    filepath = str(filepath)

    scene = BlenderScene()

    part_location = (0, 0, 0)
    angle = np.radians(90)
    part_rotation = (0, 0, 0)

    part, pv_surf, spat_dim, components = scene.add_part(sim_data=sim_data)
    scene.set_part_location(part, part_location)
    scene.set_part_rotation(part, part_rotation)


    mat_data = MaterialData()
    image_path = '/home/lorna/speckle_generator/images/blender_image_texture_rad2.tiff'
    mat = scene.add_material(mat_data, part, image_path)

    sensor_px = (2452, 2056)
    cam_position = (0, 0, 350)
    focal_length = 25
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
    render_start_time = time.perf_counter()
    image_path = Path.cwd() / 'dev/lsdev/rendered_images/case18_deformed'
    output_path = str(image_path) + '/' + name +'_report.txt'


    render_data = RenderData(samples=1, engine=RenderEngine.CYCLES)
    render = Render(render_data,
                    image_path=image_path,
                    output_path=output_path,
                    cam_data=cam_data)

    render_counter = 0
    render_name = 'ref_image'

    render.render_image(render_name, render_counter, part)

    #---------------------------------------------------------------------------
    # Deform mesh
    timesteps = sim_data.time.shape[0]
    meshdeformer = DeformMesh(pv_surf, spat_dim, components)
    nodes = centre_nodes(pv_surf.points)


    for timestep in range(1, timesteps):
        deformed_nodes = meshdeformer.add_displacement(timestep, nodes)

        if deformed_nodes is not None:
            partdeformer = DeformPart(part, deformed_nodes)
            part = partdeformer.deform_part()
            partdeformer.set_new_frame()
            print(f"{timestep=}")
            print(f"{part.dimensions=}")

            render_name = 'def_sim_data'
            render.render_image(render_name, timestep, part)
            timestep += 1 # Adding at start of loop as timestep = 0 is the original mesh


    render_end_time = time.perf_counter()
    time_render = render_end_time - render_start_time
    print('Time taken to render images: ' + str(time_render) + 's')
    report = open(output_path, 'a', encoding='utf-8')
    report.write('\nTime taken to render images: ' + str(time_render) + 's')
    report.close()

    scene.save_model(filepath)


if __name__ == "__main__":
    main()