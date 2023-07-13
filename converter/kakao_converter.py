import os
import os.path as osp
import numpy as np
from shutil import copyfile
from tqdm import tqdm
import json

from dictionary.class_dictionary import kakao_dict
from dictionary.rotation_dictionary import cam_rot, lid_rot
from scipy.spatial.transform import Rotation
from utils.util import check_valid_mat

from pyquaternion import Quaternion as Q


class kakao_parser:
    def __init__(self,
                 src_dir: str = None):
        self.src_dir = src_dir
        self.table_root = osp.join(self.src_dir, 'meta')
        self.table_names = ['dataset', 'ego_pose', 'frame', 'frame_annotation',
                            'frame_data', 'instance', 'log', 'sensor']
        self.dataset = self.__load_table__('dataset')
        self.ego_pose = self.__load_table__('ego_pose')
        self.frame = self.__load_table__('frame')
        self.frame_annotation = self.__load_table__('frame_annotation')
        self.frame_data = self.__load_table__('frame_data')
        self.instance = self.__load_table__('instance')
        self.log = self.__load_table__('log')
        self.sensor = self.__load_table__('sensor')
        self.__make_reverse_index__()

    def __load_table__(self, table_name) -> dict:
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table

    def __make_reverse_index__(self) -> None:
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            table_ = getattr(self, table)
            if isinstance(table_, dict):
                table_ = [table_]

            for ind, member in enumerate(table_):
                self._token2ind[table][member['uuid']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.frame_annotation:
            inst = self.get('instance', record['instance_uuid'])
            record['category_name'] = inst['category_name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.frame_data:
            sensor_record = self.get('sensor', record['sensor_uuid'])
            record['name'] = sensor_record['name']
            record['type'] = sensor_record['type']

        # Reverse-index samples with sample_data and annotations.
        for record in self.frame:
            record['data'] = {}
            record['anns'] = []

        for record in self.frame_data:
            if record['is_key_frame']:
                sample_record = self.get('frame', record['frame_uuid'])
                sample_record['data'][record['name']] = record['uuid']

        for ann_record in self.frame_annotation:
            frame_data_record = self.get('frame_data', ann_record['frame_data_uuid'])
            sample_record = self.get('frame', frame_data_record['frame_uuid'])
            sample_record['anns'].append(ann_record['uuid'])

    def get(self, table_name: str, token: str) -> dict:
        assert table_name in self.table_names, "Table {} not found".format(table_name)
        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        return self._token2ind[table_name][token]


def quat2euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z


class kakao:
    def __init__(self,
                 src_dir: str = None,
                 dst_dir: str = None,
                 dst_db_type: str = None):
        assert src_dir is not None or dst_dir is not None or dst_db_type is not None, \
            f'Invalid Parameter Please Check {src_dir}\n{dst_dir}\n{dst_db_type}'

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.dst_db_type = dst_db_type
        self.kakaodb = kakao_parser(self.src_dir)
        self.cam_rot = cam_rot['kakao'][dst_db_type]
        self.cam_rot = check_valid_mat(self.cam_rot)
        self.lid_rot = lid_rot['kakao'][dst_db_type]
        self.lid_rot = check_valid_mat(self.lid_rot)
        self.calib_dict = {}
        self.lid2cam = {}
        print(f'Set Destination Dataset Type {self.dst_db_type}')

    def get_corners(self, x, y, z, w, l, h, rot):
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(rot, corners)

        # Translate
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def get_projected_corners(self, points, intrinsic):
        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(intrinsic, points)
        points = points[:3, :]

        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points

    def convert(self):
        print(f'Convert kakao to {self.dst_db_type} Dataset.')

        # TODO: modify calibration data output
        for sensor in self.kakaodb.sensor:
            intrinsic = np.eye(4)
            intrinsic[:3, :3] = sensor['intrinsic']['parameter']

            loc = list(map(float, sensor['translation']))
            rot = list(map(float, sensor['rotation']))
            extrinsic = np.zeros((4, 4))
            extrinsic[:3, :3] = Q(*rot).rotation_matrix
            extrinsic[:3, 3] = loc

            self.calib_dict[sensor['name']] = {}
            self.calib_dict[sensor['name']]['intrinsic'] = intrinsic
            self.calib_dict[sensor['name']]['extrinsic'] = extrinsic

        for sensor in self.kakaodb.sensor:
            if 'lidar' not in sensor['name']:
                self.lid2cam[sensor['name']] = {}
                self.lid2cam[sensor['name']]['extrinsic'] = \
                    extrinsic @ np.linalg.inv(self.calib_dict['lidar[00]']['extrinsic']).T
                self.lid2cam[sensor['name']]['intrinsic'] = intrinsic

        for frame in tqdm(self.kakaodb.frame):
            for anno_uuid in frame['anns']:
                frame_annotation = self.kakaodb.get('frame_annotation', anno_uuid)
                frame_data = self.kakaodb.get('frame_data', frame_annotation['frame_data_uuid'])

                f_name = f'{frame_data["file_name"]}.{frame_data["file_format"]}'
                src_path = osp.join(self.src_dir, 'sensor', frame_data['name'], f_name)
                dst_path = osp.join(self.dst_dir, frame_data['type'], frame_data['name'], f_name)
                copyfile(src_path, dst_path)

                if frame_annotation['annotation_type_name'] == 'bbox_pcd3d':
                    x, y, z = list(map(float, frame_annotation['geometry']['center']))
                    w, l, h = list(map(float, frame_annotation['geometry']['wlh']))
                    _, _, yaw = quat2euler(*frame_annotation['geometry']['orientation'])
                    rot = Q(axis=[0, 0, 1], angle=-np.pi / 2).rotation_matrix

                    if self.dst_db_type == 'kitti':
                        z -= h / 2
                        yaw -= np.pi / 2

                    # project bounding box to the virtual reference frame
                    if self.dst_db_type == 'kitti':
                        x, y, z, _ = self.cam_rot @ np.array([x, y, z, 1]).T
                    else:
                        x, y, z, _ = self.lid_rot @ np.array([x, y, z, 1]).T

                    for k, v in self.lid2cam.items():
                        cam_xyz = v['extrinsic'] @ np.array([x, y, z, 1]).T
                        corners = self.get_corners(*cam_xyz, w, l, h, rot)
                        proj_mat = np.eye(4)
                        proj_mat[:3, :3] = v['intrinsic']
                        imcorners = self.get_projected_corners(corners, proj_mat)[:2]
                        print(k)
                        print(x, y, z, w, h, l, yaw)
                        print(imcorners)
                        bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))

                        # Crop bbox to prevent it extending outside image.
                        bbox_crop = tuple(max(0, b) for b in bbox)
                        bbox = (min(frame_data['width'], bbox_crop[0]),
                                min(frame_data['width'], bbox_crop[1]),
                                min(frame_data['width'], bbox_crop[2]),
                                min(frame_data['height'], bbox_crop[3]))

                        # Detect if a cropped box is empty.
                        if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
                            bbox = (0, 0, 0, 0)

                        x1, y1, x2, y2 = bbox

                        if bbox == (0, 0, 0, 0):
                            continue

                        with open(
                                f'{osp.join(self.dst_dir, "label", k, frame_data["file_name"] + ".txt")}',
                                'a') as f:
                            if self.dst_db_type == 'kitti' and yaw != 'None':
                                cls = kakao_dict[f'to_kitti'][frame_annotation['category_name']]
                                f.write(
                                    f'{cls}, 0, 0, -10, {x1}, {y1}, {x2}, {y2}, {x}, {y}, {z}, {h}, {w}, {l}, {yaw}\n')
                            elif self.dst_db_type == 'waymo':
                                cls = kakao_dict[f'to_waymo'][frame_annotation['category_name']]
                                w = x2 - x1
                                h = y2 - y1
                                cx = x1 + w / 2
                                cy = y1 + h / 2
                                f.write(f'{cx}, {cy}, {w}, {h}, 0, 0, {cls}, -1, '
                                        f'{x}, {y}, {z}, {w}, {l}, {h}, {yaw}\n')
                            elif self.dst_db_type == 'nuscenes':
                                cls = kakao_dict[f'to_nuscenes'][frame_annotation['category_name']]
                                rot = Rotation.from_euler('xyz', [0, 0, rot])
                                rot_quat = rot.as_quat()
                                f.write(f'{cls}, {x}, {y}, {z}, {w}, {h}, {l}, '
                                        f'{rot_quat[0]}, {rot_quat[1]}, {rot_quat[2]}, {rot_quat[3]}, '
                                        f'0, 0, {x1}, {y1}, {x2}, {y2}\n')
                            elif self.dst_db_type == 'udacity':
                                cls = kakao_dict[f'to_udacity'][frame_annotation['category_name']]
                                f.write(f'{x1}, {y1}, {x2}, {y2}, {cls}\n')
