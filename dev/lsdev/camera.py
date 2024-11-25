from dataclasses import dataclass
import numpy as np
import bpy

@dataclass
class CameraData:
    xpix : int | None = 2452
    ypix : int | None = 2056
    position : tuple | None = (0, 0, 10)
    orientation : tuple | None = (0, 0, 0)
    object_distance : float | None = None
    fstop: float | None = 0
    focal_length : float | None = 50.0
    sensor_size : tuple | None = (8.4594, 7.0932)
    sensor_px : tuple | None = (xpix, ypix)
    k1 : float | None = 0.0
    k2 : float | None = 0.0
    k3 : float | None = 0.0
    p1 : float | None = 0.0
    p2 : float | None = 0.0
    c0 : float | None = None
    c1 : float | None = None

# TODO: write get and set methods for data class variables

class CameraBlender():
    def __init__(self):
        self.camera_data = CameraData

    def add_camera(self):
        new_cam = bpy.data.cameras.new('Camera')
        camera = bpy.data.objects.new('Camera', new_cam)
        bpy.context.collection.objects.link(camera)

        camera.location = self.camera_data.position
        camera.rotation_mode = 'XYZ' # TODO: Make this a variable with diff options
        camera.rotation_euler = self.camera_data.orientation

        camera['sensor_px'] = self.camera_data.sensor_px
        camera['px_size'] = [i / j for i, j in zip(self.camera_data.sensor_size,
                                                    self.camera_data.sensor_px)]
        camera['k1'] = self.camera_data.k1
        camera['k2'] = self.camera_data.k2
        camera['k3'] = self.camera_data.k3
        camera['p1'] = self.camera_data.p1
        camera['p2'] = self.camera_data.p2

        if self.camera_data.c0 is None:
            camera['c0'] = (self.camera_data.sensor_px[0]) / 2
        else:
            camera['c0'] = self.camera_data.c0
        if self.camera_data.c1 is None:
            camera['c1'] = (self.camera_data.sensor_px[1]) / 2
        else:
            camera['c1'] = self.camera_data.c1

        new_cam.lens = self.camera_data.focal_length
        new_cam.sensor_width = self.camera_data.sensor_size[0]
        new_cam.sensor_height = self.camera_data.sensor_size[1]

        if self.camera_data.object_distance is not None:
            new_cam.dof.focus_distance = self.camera_data.object_distance
            new_cam.dof.use_dof = True
            new_cam.dof.aperture_fstop = self.camera_data.fstop

        bpy.context.scene.camera = camera

        return camera

    def stereo_setup(self,
                     stereo_angle: float,
                     orientation1,):
        # TODO: Write method to create stereo setup given params
        self.add_camera()
        self.set_orientation()
        self.add_camera()


