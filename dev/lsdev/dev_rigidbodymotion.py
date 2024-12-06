import bpy
from dev.lsdev.dev_render import RenderData, Render

class RigidBodyMotion:
    def __init__(self, part, step, part_location, image_path, output_path):
        self.part = part
        self.step = step
        self.part_location = part_location
        self.image_path = image_path
        self.output_path = output_path

    def rigid_body_motion_z(self, z_lims):
        min_z = z_lims[0]
        max_z = z_lims[1]

        n_steps = (max_z - min_z) / self.step
        render_counter = 0

        # TODO: Make this loop work with int + float values

        for z in range(n_steps):
            z_location = (z * self.step) + min_z
            print(f"{z_location=}")
            self.part.location = (self.part_location[0],
                                  self.part_location[1],
                                  (self.part_location[2] + z_location))
            render_data = RenderData(samples=1)
            render = Render(render_data, self.image_path, self.output_path)

            render_name = 'rigid_body_motion_z'
            render.render_image(render_name, render_counter)
            render_counter += 1

    def rigid_body_motion_x(self, x_lims):
        min_x = x_lims[0]
        max_x = x_lims[1]

        n_steps = int((max_x - min_x) / self.step)
        render_counter = 0

        for x in range(n_steps):
            x_location = (x * self.step) + min_x
            print(f"{x_location=}")
            self.part.location = ((self.part_location[0] + x_location),
                                  self.part_location[1],
                                  self.part_location[2])
            render_data = RenderData(samples=1)
            render = Render(render_data, self.image_path, self.output_path)

            render_name = 'rigid_body_motion_x'

            render.render_image(render_name, render_counter)
            render_counter += 1
