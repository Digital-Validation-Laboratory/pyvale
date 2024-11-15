from dataclasses import dataclass
import numpy as np
import bpy

@dataclass
class CameraData:
    xpix : int | None = None
    ypix : int | None = None
    position : np.ndarray | None = None
    orientation : np.ndarray | None = None
    object_distance : float | None = None
    fstop: float | None = None
    focal_length : float | None = None
    sensor_size : np.ndarray | None = None
    sensor_px : np.ndarray | None = None
    k1 : float | None = None
    k2 : float | None = None
    k3 : float | None = None
    p1 : float | None = None
    p2 : float | None = None
    c0 : float | None = None
    c1 : float | None = None

class CameraBlender():
    def __init__(self):
        self.camera_data = CameraData

    def add_camera(self):
        new_cam = bpy.data.cameras.new('Camera')
        camera = bpy.data.objects.new('Camera', new_cam)
        bpy.context.collection.objects.link(camera)

        camera.location = self.camera_data.position
        camera.rotation_mode = 'EULER' # TODO: Make this a variable with diff options
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

        self.cameras.append(camera) # not sure if need this
        return camera







    def stereo_setup():
        # TODO: Write method to create stereo setup given params
