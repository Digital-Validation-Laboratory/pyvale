import numpy as np
from pathlib import Path
import mooseherder as mh
from mooseherder import SimData
from dev_partblender import BlenderPart

class DeformMesh:
    def __init__(self, sim_data:SimData, defgrad):
        self.sim_data = sim_data
        self.defgrad = defgrad
        self.nodes = self._get_nodes()

    def _get_nodes(self):
        mesh_builder = BlenderPart(self.sim_data)
        nodes = mesh_builder._get_nodes()
        return nodes

    def map_coords(self):
        mid = np.max(self.nodes, axis=0)  / 2
        centred = np.subtract(self.nodes, mid)

        defgrad_inv = np.linalg.inv(self.defgrad)

        coords_new = np.einsum('ij,nj->ni', defgrad_inv, centred)

        coords_uncentred = coords_new + mid

        return coords_uncentred



