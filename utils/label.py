import numpy as np
from .util import bbox_2d, bbox_3d


class Label:
    def __init__(self, class_name: str,
                 left: int, top: int, right: int, bottom: int,
                 x: float, y: float, z: float,
                 width: float, height: float, length: float,
                 rot: float):
        self.class_name = class_name
        self.label_2d = bbox_2d(left, top, right, bottom)
        self.label_3d = bbox_3d(x, y, z, width, height, length, rot)

    def get_coords(self):
        x, y, z = self.label_3d.get_coords()
        return np.array([x, y, z])

    def set_coords(self, x: float, y: float, z: float):
        self.label_3d.set_coords(x, y, z)

    def get_dims(self):
        w, h, l = self.label_3d.get_dims()
        return np.array([w, h, l])

    def set_dims(self, w: float, h: float, l: float):
        self.label_3d.set_dims(w, h, l)

    def get_rot(self):
        return self.label_3d.rot

    def set_rot(self, rot: float):
        self.label_3d.set_rot(rot)
