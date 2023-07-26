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
import open3d as o3d
import cv2


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
        print(f'Set Destination Dataset Type {self.dst_db_type}')
        if 'like' in dst_db_type:
            dst_db_type = 'kitti'
        self.cam_rot = cam_rot['kakao'][dst_db_type]
        self.cam_rot = check_valid_mat(self.cam_rot)
        self.lid_rot = lid_rot['kakao'][dst_db_type]
        self.lid_rot = check_valid_mat(self.lid_rot)
        self.calib_dict = {}
        self.img_size = {}

    def get_corners(self, x, y, z, w, l, h, rot):
        # 3D bounding box corners. (Convention: x points right, y to the down, z forward)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = h / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = w / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
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
        camera_names = ['camera(00)', 'camera(01)', 'camera(02)', 'camera(03)', 'camera(04)', 'camera(05)']

        for idx, frame in enumerate(tqdm(self.kakaodb.frame[:100])):
            # KAKAO dataset LiDAR model : HESAI Pandar128
            frame_data_uuid = frame['data']['lidar(00)']
            frame_data = self.kakaodb.get('frame_data', frame_data_uuid)
            src_path = f'{self.src_dir}sensor/{frame_data["name"]}/{frame_data["file_name"]}.{frame_data["file_format"]}'
            dst_path = f'{self.dst_dir}{frame_data["type"]}/{frame_data["name"]}/{idx:06d}.{frame_data["file_format"]}'

            if frame_data["file_format"] == 'pcd':
                pcd = o3d.io.read_point_cloud(src_path)
                points = np.asarray(pcd.points)
                if 'like' not in self.dst_db_type:
                    points = self.lid_rot[:3, :3] @ points.T
                    points = points.T
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(dst_path, pcd)
            else:
                # N * [x, y, z, intensity, time]
                points = np.fromfile(src_path, dtype=np.float32).reshape(5, -1).T[:, :4]
                intensity = points[:, 3]
                if 'like' not in self.dst_db_type:
                    points = self.lid_rot[:3, :3] @ points[:, :3].T
                    points = points.T
                    points[:, 3] = intensity
                points.astype(np.float32).tofile(dst_path)

            sensor_data = self.kakaodb.get('sensor', frame_data['sensor_uuid'])
            loc = list(map(float, sensor_data['translation']))
            rot = list(map(float, sensor_data['rotation']))
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = Q(*rot).rotation_matrix
            extrinsic[:3, 3] = loc

            if 'like' not in self.dst_db_type:
                lid_extrinsic = self.lid_rot @ extrinsic
            else:
                lid_extrinsic = extrinsic

            for camera_name in camera_names:
                frame_data_uuid = frame['data'][camera_name]
                frame_data = self.kakaodb.get('frame_data', frame_data_uuid)
                src_path = f'{self.src_dir}sensor/{frame_data["name"]}/{frame_data["file_name"]}.{frame_data["file_format"]}'
                dst_path = f'{self.dst_dir}{frame_data["type"]}/{frame_data["name"]}/{idx:06d}.{frame_data["file_format"]}'
                copyfile(src_path, dst_path)

                img_shape = cv2.imread(src_path).shape
                self.img_size[camera_name] = (img_shape[1], img_shape[0])

                sensor_data = self.kakaodb.get('sensor', frame_data['sensor_uuid'])
                k = np.zeros((3, 3))

                p = np.zeros((3, 4))
                p[:3, :3] = sensor_data['intrinsic']['parameter']

                d = np.zeros((1, 5))

                loc = list(map(float, sensor_data['translation']))
                rot = list(map(float, sensor_data['rotation']))
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = Q(*rot).rotation_matrix
                extrinsic[:3, 3] = loc

                self.calib_dict[sensor_data['name']] = {}
                self.calib_dict[sensor_data['name']]['p'] = p
                self.calib_dict[sensor_data['name']]['extrinsic'] = extrinsic

                with open(f'{self.dst_dir}calib/{camera_name}/{idx:06d}.txt', 'w') as f:
                    if self.dst_db_type == 'nuscenes':
                        f.write(f'{camera_name}_intrinsic : '
                                f'{", ".join(list(map(str, self.calib_dict[camera_name]["P"].flatten())))}\n')
                        cam_extrinsic = self.cam_rot @ self.calib_dict[camera_name]["extrinsic"]
                        f.write(f'{camera_name}_extrinsic : '
                                f'{", ".join(list(map(str, cam_extrinsic.flatten())))}\n')
                        f.write(f'lidar(00)_extrinsic : '
                                f'{", ".join(list(map(str, lid_extrinsic.flatten())))}\n')
                    elif self.dst_db_type == 'waymo':
                        f.write(f'{camera_name}_intrinsic : '
                                f'{", ".join(list(map(str, self.calib_dict[camera_name]["P"].flatten())))}\n')
                        cam_extrinsic = self.cam_rot @ self.calib_dict[camera_name]["extrinsic"]
                        f.write(f'{camera_name}_extrinsic : '
                                f'{", ".join(list(map(str, cam_extrinsic.flatten())))}\n')
                        f.write(f'{camera_name}_width : 1920\n')
                        f.write(f'{camera_name}_height : 1200\n')
                        f.write(f'{camera_name}_rolling_shutter_direction : -1\n')
                        f.write(f'lidar(00)_beam_inclinations : -1\n')
                        f.write(f'lidar(00)_beam_inclination_min : -1\n')
                        f.write(f'lidar(00)_beam_inclination_max : -1\n')
                        f.write(f'lidar(00)_extrinsic : '
                                f'{", ".join(list(map(str, lid_extrinsic.flatten())))}\n')
                    elif 'kitti' in self.dst_db_type:
                        f.write(f'K: '
                                f'{", ".join(list(map(str, k.flatten())))}\n')
                        f.write(f'P: '
                                f'{", ".join(list(map(str, self.calib_dict[camera_name]["p"].flatten())))}\n')
                        f.write(f'D: '
                                f'{", ".join(list(map(str, d.flatten())))}\n')
                        cam_extrinsic = self.cam_rot @ self.calib_dict[camera_name]["extrinsic"]
                        lid2cam = np.linalg.inv(cam_extrinsic) @ lid_extrinsic
                        f.write(f'Tr_velo_to_cam: '
                                f'{", ".join(list(map(str, lid2cam.flatten())))}\n')
                        f.write(f'Tr_imu_to_cam: '
                                f'{", ".join(list(map(str, cam_extrinsic.flatten())))}\n')

            no_label_sensor_name = set()

            for anno_uuid in frame['anns']:
                frame_annotation = self.kakaodb.get('frame_annotation', anno_uuid)

                if frame_annotation['annotation_type_name'] == 'bbox_pcd3d':
                    x, y, z = list(map(float, frame_annotation['geometry']['center']))
                    w, l, h, = list(map(float, frame_annotation['geometry']['wlh']))
                    yaw_, _, _ = Q(*frame_annotation['geometry']['orientation']).yaw_pitch_roll

                    if 'like' in self.dst_db_type:
                        yaw = yaw_ + np.pi / 2
                        if not os.path.exists(f'{self.dst_dir}label/lidar(00)'):
                            os.makedirs(f'{self.dst_dir}label/lidar(00)')

                        with open(f'{self.dst_dir}label/lidar(00)/{idx:06d}.txt', 'a') as f:
                            f.write(f'{frame_annotation["category_name"]}, -1, 3, -99, -1, -1, -1, -1, '
                                    f'{h:.4f}, {w:.4f}, {l:.4f}, {x:.4f}, {y:.4f}, {z:.4f}, {yaw:.4f}, -1, -1\n')

                    for camera_name in camera_names:
                        rt_mat = np.linalg.inv(self.calib_dict[camera_name]['extrinsic']) @ lid_extrinsic
                        cam_x, cam_y, cam_z, _ = rt_mat @ np.array([x, y, z, 1]).T
                        cam_yaw, _, _ = Q(matrix=np.linalg.inv(self.calib_dict[camera_name]['extrinsic'][:3, :3])).yaw_pitch_roll
                        yaw = -yaw_ - cam_yaw

                        if cam_z < 0: continue
                        corners = self.get_corners(cam_x, cam_y, cam_z, w, l, h,
                                                   Q(axis=(0, 1, 0), angle=yaw).rotation_matrix)
                        imcorners = self.get_projected_corners(corners, self.calib_dict[camera_name]['p'])[:2]

                        if np.all(abs(imcorners[0]) > self.img_size[camera_name][0]) or \
                                np.all(abs(imcorners[1]) > self.img_size[camera_name][1]): continue
                        bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))

                        # Crop bbox to prevent it extending outside image.
                        bbox_crop = tuple(max(0, b) for b in bbox)

                        # Detect if a cropped box is empty.
                        if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
                            continue

                        bbox = (min(self.img_size[camera_name][0], bbox_crop[0]),
                                min(self.img_size[camera_name][0], bbox_crop[1]),
                                min(self.img_size[camera_name][0], bbox_crop[2]),
                                min(self.img_size[camera_name][1], bbox_crop[3]))

                        if bbox == (0, 0, self.img_size[camera_name][0], self.img_size[camera_name][1]):
                            continue

                        x1, y1, x2, y2 = bbox

                        if 'like' not in self.dst_db_type:
                            cls = kakao_dict[f'to_{self.dst_db_type}'][frame_annotation['category_name']]
                        else:
                            cls = frame_annotation['category_name']

                        no_label_sensor_name.add(camera_name)

                        with open(f'{self.dst_dir}label/{camera_name}/{idx:06d}.txt', 'a') as f:
                            if self.dst_db_type != 'udacity':
                                if 'kitti' not in self.dst_db_type:
                                    x, y, z = np.dot(self.lid_rot, np.array([x, y, z, 1]).T)

                                if 'kitti' in self.dst_db_type:
                                    cam_y += h / 2
                                    f.write(f'{cls}, -1, 3, -99, '
                                            f'{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}, '
                                            f'{h:.4f}, {w:.4f}, {l:.4f}, {cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f}, {yaw:.4f}, -1, -1\n')
                                elif self.dst_db_type == 'waymo':
                                    width = x2 - x1
                                    cx = x1 + width / 2
                                    height = y2 - y1
                                    cy = y1 + height / 2
                                    f.write(f'{cx:.4f}, {cy:.4f}, {width:.4f}, {height:.4f}, 0, 0, {cls:.4f}, -1, '
                                            f'{x:.4f}, {y:.4f}, {z:.4f}, {w:.4f}, {l:.4f}, {h:.4f}, {yaw:.4f}\n')
                                elif self.dst_db_type == 'nuscenes':
                                    rot = Rotation.from_euler('xyz', [0, 0, rot])
                                    rot_quat = rot.as_quat()
                                    f.write(f'{cls}, {x:.4f}, {y:.4f}, {z:.4f}, {w:.4f}, {h:.4f}, {l:.4f}, '
                                            f'{rot_quat[0]:.4f}, {rot_quat[1]:.4f}, {rot_quat[2]:.4f}, {rot_quat[3]:.4f}, '
                                            f'0, 0, {x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}\n')
                            else:
                                f.write(f'{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}, {cls}\n')

            for camera_name in camera_names:
                if camera_name not in no_label_sensor_name:
                    with open(f'{self.dst_dir}label/{camera_name}/{idx:06d}.txt', 'a') as f:
                        f.write('')
