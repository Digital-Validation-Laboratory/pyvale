import bpy

class BlenderLight():
    def __init__(self):
        self.type
        self.light_ob
        self.light

    def add_light():
        light = bpy.data.lights.new(name='spot', type='SPOT')
        light_ob = bpy.data.objects.new(name='Spot', object_data=light)
        bpy.context.collection.objects.link(light_ob)

    def get_light(self):
        self.light_ob = light_ob

    def set_location(self):
        self.slight_ob.location = pos

    def set_rotation(self):
        self.light_ob.rotation_mode = 'QUATERNION'
        self.light_ob.rotation_quaternion = orientation

    def set_energy(self):
        self.light = energy

