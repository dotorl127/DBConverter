import os
import numpy as np
from shutil import copyfile
from tqdm import tqdm

from dictionary.class_dictionary import kitti_dict
from dictionary.rotation_dictionary import cam_rot, lid_rot
from scipy.spatial.transform import Rotation
from utils import label
from utils.util import check_valid_mat


class kitti_label(label.Label):
    def __init__(self, class_name: str, truncated: float, occluded: int, alpha: float,
                 left: int, top: int, right: int, bottom: int, x: float, y: float, z: float,
                 width: float, height: float, length: float, rotation_y: float):
        super().__init__(class_name, left, top, right, bottom, x, y, z, width, height, length, rotation_y)
        self.truncated = truncated
        self.occluded = occluded
        self.alpha = alpha


class kitti:
    def __init__(self,
                 src_dir: str = None,
                 dst_dir: str = None,
                 dst_db_type: str = None):
        """
        :param src_dir: something.
        :param dst_dir: something.
        """
        assert src_dir is not None or dst_dir is not None or dst_db_type is not None, \
            f'Invalid Parameter Please Check {src_dir}\n{dst_dir}\n{dst_db_type}'

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.dst_db_type = dst_db_type
        self.calib_dict = {}
        self.image = None
        self.labels = []
        self.points = None
        self.cam_rot = cam_rot['kitti'][dst_db_type]
        self.cam_rot = check_valid_mat(self.cam_rot)
        self.lid_rot = lid_rot['kitti'][dst_db_type]
        self.lid_rot = check_valid_mat(self.lid_rot)
        self.rt_mat = np.eye(4)
        print(f'Set Destination Dataset Type {self.dst_db_type}')

    def label_convert(self, label: kitti_label, dst_db_type: str):
        type = kitti_dict[f'to_{dst_db_type}'][label.class_name]
        if dst_db_type == 'waymo':
            w = int(label.label_2d.x2 - label.label_2d.x1)
            h = int(label.label_2d.y2 - label.label_2d.y1)
            cnt_x = int(label.label_2d.x1 + w / 2)
            cnt_y = int(label.label_2d.y1 + h / 2)
            return f'{cnt_x}, {cnt_y}, {w}, {h}, 0, 0, {type}, -1, ' \
                   f'{label.label_3d.locs[0]}, {label.label_3d.locs[1]}, {label.label_3d.locs[2]}, ' \
                   f'{label.label_3d.dims[0]}, {label.label_3d.dims[1]}, {label.label_3d.dims[2]}, ' \
                   f'{label.label_3d.rot}, 0'
        else:
            rot = Rotation.from_euler('xyz', [0, 0, label.label_3d.rot])
            rot_quat = rot.as_quat()
            return f'{type}, {label.label_3d.locs[0]}, {label.label_3d.locs[1]}, {label.label_3d.locs[2]}, ' \
                   f'{label.label_3d.dims[0]}, {label.label_3d.dims[1]}, {label.label_3d.dims[2]}, ' \
                   f'{rot_quat[0]}, {rot_quat[1]}, {rot_quat[2]}, {rot_quat[3]}, {-1}, {-1}, ' \
                   f'{label.label_2d.x1}, {label.label_2d.y1}, {label.label_2d.x2}, {label.label_2d.y2}'

    def convert(self):
        print(f'Convert Kitti to {self.dst_db_type} Dataset.')

        calib_path = self.src_dir + 'calib/'
        img_path = self.src_dir + 'image_2/'
        label_path = self.src_dir + 'label_2/'
        lid_path = self.src_dir + 'velodyne/'
        filenames = sorted(os.listdir(calib_path))
        filenames = [filename.rstrip('.txt') for filename in filenames]

        calib_filter = ['P2', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo']

        for index, filename in enumerate(tqdm(filenames[:10])):
            self.calib_dict.clear()
            # calibration file read, parse
            with open(f'{calib_path}{filename}.txt') as f:
                lines = f.readlines()
                for line in lines:
                    token = line.strip().split(':')
                    if token[0] in calib_filter:
                        self.calib_dict[token[0]] = np.array(token[1].strip().split(' ')).astype(float).reshape(3, -1)

            # origin image path read
            self.image = f'{img_path}{filename}.png'

            self.labels.clear()
            # label file read, parse
            with open(f'{label_path}{filename}.txt') as f:
                lines = f.readlines()
                for line in lines:
                    token = line.strip().split(' ')
                    # passing invalid label
                    if float(token[-1]) == -10:
                        continue
                    # class_name, truncated, occluded, 2d label(left, top, right, bottom),
                    # dimension(height, width, length), location(x, y, z), rotation_y
                    label = kitti_label(token[0], float(token[1]), int(token[2]), float(token[3]),
                                        int(float(token[4])), int(float(token[5])),
                                        int(float(token[6])), int(float(token[7])),
                                        float(token[11]), float(token[12]), float(token[13]),
                                        float(token[9]), float(token[8]), float(token[10]),
                                        float(token[14]))
                    self.labels.append(label)

            # origin point cloud file read, parse
            self.points = np.fromfile(f'{lid_path}{filename}.bin', dtype=np.float32).reshape(-1, 4)

            # calibration file value convert valid
            for name, mat in self.calib_dict.items():
                # P{X}, R{X}_rect is projection, rectifying matrix so doesn't do any calculating
                if name == 'Tr_velo_to_cam':
                    mat = check_valid_mat(mat)
                    self.calib_dict[name] = mat
                elif name == 'Tr_imu_to_velo':
                    mat = check_valid_mat(mat)
                    self.calib_dict[name] = mat
                else:
                    mat = check_valid_mat(mat)
                    self.calib_dict[name] = mat

            # make RT matrix to convert coordinates system
            self.calib_dict['Tr_cam_to_imu'] = \
                np.linalg.inv(self.calib_dict['Tr_imu_to_velo']) @ \
                np.linalg.inv(self.cam_rot @ self.calib_dict['Tr_velo_to_cam']) @ \
                np.linalg.inv(self.calib_dict['R0_rect'])
            self.rt_mat = self.lid_rot @ np.linalg.inv(self.calib_dict['Tr_velo_to_cam'])

            with open(f'{self.dst_dir}calib/image_2/{index:06d}.txt', 'w') as f:
                if self.dst_db_type == 'waymo':
                    intrinsic = self.calib_dict['P2']
                    line = ', '.join(map(str, intrinsic.reshape(-1).tolist())) + '\n'
                    f.write(f'image_2_intrinsic: {line}')
                    line = ', '.join(map(str, self.calib_dict['Tr_cam_to_imu'].reshape(-1).tolist())) + '\n'
                    f.write(f'image_2_extrinsic: {line}')
                    f.write(f'image_2_width: 1224\n')
                    f.write(f'image_2_height: 370\n')
                    f.write(f'image_2_rolling_shutter_direction: -1\n')
                    f.write(f'velodyne_beam_inclinations: -1\n')
                    f.write(f'velodyne_beam_inclination_min: -1\n')
                    f.write(f'velodyne_beam_inclination_max: -1\n')
                    line = ', '.join(map(str, np.linalg.inv(self.lid_rot @
                                                            self.calib_dict['Tr_imu_to_velo']).reshape(-1).tolist())) + '\n'
                    f.write(f'velodyne_extrinsic: {line}')
                else:
                    line = ', '.join(map(str, self.calib_dict['Tr_cam_to_imu'][:3, 3].reshape(-1).tolist())) + '\n'
                    f.write(f'image_2_translation: {line}')
                    rot = Rotation.from_matrix(self.calib_dict['Tr_cam_to_imu'][:3, :3])
                    line = ', '.join(map(str, rot.as_quat().reshape(-1).tolist())) + '\n'
                    f.write(f'image_2_rotation: {line}')
                    intrinsic = self.calib_dict['P2']
                    line = ', '.join(map(str, intrinsic.reshape(-1).tolist())) + '\n'
                    f.write(f'image_2_intrinsic: {line}')

                    line = ', '.join(map(str, np.linalg.inv(self.lid_rot @
                                                            self.calib_dict['Tr_imu_to_velo'])[:, 3].reshape(-1).tolist())) + '\n'
                    f.write(f'velodyne_translation: {line}')
                    rot = Rotation.from_matrix(np.linalg.inv(self.calib_dict['Tr_imu_to_velo'])[:3, :3])
                    line = ', '.join(map(str, rot.as_quat().reshape(-1).tolist())) + '\n'
                    f.write(f'velodyne_rotation: {line}')

            copyfile(self.image, f'{self.dst_dir}camera/image_2/{index:06d}.png')

            # convert label
            with open(f'{self.dst_dir}label/image_2/{index:06d}.txt', 'w') as f:
                for label in self.labels:
                    x, y, z = label.get_coords()
                    w, h, l = label.get_dims()
                    y -= h / 2
                    x, y, z, _ = self.rt_mat @ np.array([x, y, z, 1])
                    label.set_coords(x, y, z)
                    if self.dst_db_type == 'nuscenes':
                        rot = label.get_rot()
                        label.set_rot(-rot - np.pi / 2)
                    else:
                        rot = label.get_rot()
                        label.set_rot(-rot)
                    line = self.label_convert(label, self.dst_db_type) + '\n'
                    f.write(line)

            # convert point cloud
            intensity = self.points[:, 3]
            self.points = (self.lid_rot @ self.points.T).T
            self.points[:, 3] = intensity
            self.points.astype(np.float32).tofile(f'{self.dst_dir}lidar/velodyne/{index:06d}.bin')
