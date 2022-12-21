import numpy as np
from scipy.spatial.transform import Rotation
from pyquaternion.quaternion import Quaternion as Q


class bbox_2d:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_coords(self):
        return self.x1, self.y1, self.x2, self.y2


class bbox_3d:
    def __init__(self, x: float, y: float, z: float, width: float, height: float, length: float, rot: float):
        self.locs = (x, y, z)
        self.dims = (width, height, length)
        self.rot = rot

    def get_coords(self):
        return self.locs

    def set_coords(self, x: float, y: float, z: float):
        self.locs = (x, y, z)

    def get_dims(self):
        return self.dims

    def set_dims(self, w: float, h: float, l: float):
        self.dims = (w, h, l)

    def get_rot(self):
        return self.rot

    def set_rot(self, rot: float):
        self.rot = rot


def check_valid_mat(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] != 4 or mat.shape[1] != 4:
        temp = np.eye(4, dtype=np.float32)
        temp[:mat.shape[0], :mat.shape[1]] = mat[:mat.shape[0], :mat.shape[1]]
        return temp
    return mat


def parse_label(dataset_type: str, label: list):
    label_2d = None
    label_3d = None
    label_cls = None

    # label_2d : [left, top, rigth, bottom]
    # label_3d : [x, y, z, width, length, height, rot]

    if dataset_type == 'kitti':
        label_2d = [*list(map(int, label[4:8]))]
        x, y, z = list(map(float, label[11:14]))
        rot = np.linalg.inv(Q(axis=[1, 0, 0], angle=np.pi / 2).rotation_matrix @
                            Q(axis=[0, 0, 1], angle=np.pi / 2).rotation_matrix)
        rot = check_valid_mat(rot)
        x, y, z, _ = rot @ np.array([x, y, z, 1])
        label_3d = [x, y, z + float(label[8]) / 2,
                    float(label[9]), float(label[10]), float(label[8]), float(label[14])]
        label_cls = label[0]
    elif dataset_type == 'waymo':
        x1 = int(label[0]) - int(int(label[2]) / 2)
        y1 = int(label[1]) - int(int(label[3]) / 2)
        x2 = x1 + int(label[2])
        y2 = y1 + int(label[3])
        label_2d = [x1, y1, x2, y2]
        label_3d = [*list(map(float, label[8:11])),
                    float(label[11]), float(label[12]), float(label[13]), float(label[14])]
        label_cls = label[6]
    elif dataset_type == 'nuscenes':
        label_2d = list(map(int, label[13:]))
        if list(map(float, label[7:11])) != [0, 0, 0, 0]:
            rot = Rotation.from_quat(list(map(float, label[7:11])))
            rot_z = rot.as_euler('xyz')[-1]
            label_3d = [*list(map(float, label[1:4])),
                        float(label[4]), float(label[6]), float(label[5]), rot_z]
        label_cls = label[0]
    elif dataset_type == 'udacity':
        label_2d = [*list(map(int, label[:-1]))]
        label_cls = label[-1]

    return label_2d, label_3d, label_cls
