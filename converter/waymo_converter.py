import os.path
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from pyquaternion import Quaternion as Q

from dictionary.class_dictionary import waymo_dict
from dictionary.rotation_dictionary import cam_rot, lid_rot
from scipy.spatial.transform import Rotation
from utils.util import check_valid_mat

import tensorflow as tf
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2 as W
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils


MERGE_POINT_FLAG = False


class waymo:
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
        self.cam_rot_dict = {}
        print(f'Set Destination Dataset Type {dst_db_type}')
        if 'kitti' in dst_db_type:
            dst_db_type = 'kitti'
        self.cam_rot = cam_rot['waymo'][dst_db_type]
        self.cam_rot = check_valid_mat(self.cam_rot)
        self.lid_rot = lid_rot['waymo'][dst_db_type]
        self.lid_rot = check_valid_mat(self.lid_rot)
        self.rt_mat = np.eye(4)
        self.int_to_cam_name = {0: 'UNKNOWN',
                                1: 'FRONT',
                                2: 'FRONT_LEFT',
                                3: 'FRONT_RIGHT',
                                4: 'SIDE_LEFT',
                                5: 'SIDE_RIGHT'}
        self.int_to_lid_name = {0: 'UNKNOWN',
                                1: 'TOP',
                                2: 'FRONT',
                                3: 'SIDE_LEFT',
                                4: 'SIDE_RIGHT',
                                5: 'REAR'}
        self.int_to_class_name = list(waymo_dict[f'to_{dst_db_type}'].values())

    def calib_convert(self, frame, idx: int):
        lid_extrinsic = ''

        for lidar in frame.context.laser_calibrations:
            if lidar.name == 1:
                lid_extrinsic = np.array(lidar.extrinsic.transform).reshape(4, 4)
                break
            else:
                continue

        for camera in frame.context.camera_calibrations:
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(4, 4)

            cam_intrinsic = np.zeros((4, 4))
            cam_intrinsic[0, 0] = camera.intrinsic[0]
            cam_intrinsic[1, 1] = camera.intrinsic[1]
            cam_intrinsic[0, 2] = camera.intrinsic[2]
            cam_intrinsic[1, 2] = camera.intrinsic[3]
            cam_intrinsic[2, 2] = 1

            with open(f'{self.dst_dir}calib/{self.int_to_cam_name[camera.name]}/{idx:06d}.txt', 'w') as f:
                if 'kitti' in self.dst_db_type:
                    Tr_velo_to_cam = self.cam_rot @ np.linalg.inv(T_cam_to_vehicle)
                    self.cam_rot_dict[int(camera.name)] = Tr_velo_to_cam
                    Tr_imu_to_velo = self.lid_rot @ np.linalg.inv(lid_extrinsic)

                    line = ', '.join(map(str, cam_intrinsic.reshape(-1).tolist())) + '\n'
                    f.write(f'P: {line}')
                    line = ', '.join(map(str, np.eye(3).reshape(-1).tolist())) + '\n'
                    f.write(f'R0_rect: {line}')
                    line = ', '.join(map(str, Tr_velo_to_cam.reshape(-1).tolist())) + '\n'
                    f.write(f'Tr_velo_to_cam: {line}')
                    line = ', '.join(map(str, Tr_imu_to_velo.reshape(-1).tolist())) + '\n'
                    f.write(f'Tr_imu_to_velo: {line}')
                else:
                    line = ', '.join(map(str, (self.cam_rot @ T_cam_to_vehicle)[:3, 3].reshape(-1).tolist())) + '\n'
                    f.write(f'{self.int_to_cam_name[camera.name]}_translation: {line}')
                    rotation = Rotation.from_matrix((self.cam_rot @ T_cam_to_vehicle)[:3, :3])
                    line = ', '.join(map(str, rotation.as_quat().reshape(-1).tolist())) + '\n'
                    f.write(f'{self.int_to_cam_name[camera.name]}_rotation: {line}')
                    line = ', '.join(map(str, cam_intrinsic.reshape(-1).tolist())) + '\n'
                    f.write(f'{self.int_to_cam_name[camera.name]}_intrinsic: {line}')
                    line = ', '.join(map(str, lid_extrinsic[:3, 3].reshape(-1).tolist())) + '\n'
                    f.write(f'TOP_translation: {line}')
                    rotation = Rotation.from_matrix(lid_extrinsic[:3, :3])
                    line = ', '.join(map(str, rotation.as_quat().reshape(-1).tolist())) + '\n'
                    f.write(f'TOP_rotation: {line}')

    def save_image(self, frame, idx: int):
        for img in frame.images:
            img_path = f'{self.dst_dir}camera/{self.int_to_cam_name[img.name]}/{idx:06d}.png'
            img = cv2.imdecode(np.frombuffer(img.image, np.uint8), cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.imsave(img_path, rgb_img, format='png')

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0,
                                           merge_points=True):
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points = {}
        cp_points = {}
        intensity = {}

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            # No Label Zone
            nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
            range_image_mask = range_image_mask & nlz_mask
            if MERGE_POINT_FLAG:
                extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
            else:
                extrinsic = np.eye(4)

            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.compat.v1.where(range_image_mask))

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor,
                                            tf.compat.v1.where(range_image_mask))

            points[c.name] = points_tensor.numpy()
            cp_points[c.name] = cp_points_tensor.numpy()

            intensity_tensor = tf.gather_nd(range_image_tensor,
                                            tf.where(range_image_mask))
            intensity[c.name] = intensity_tensor.numpy()[:, 1]
        return points, cp_points, intensity

    def save_lidar(self, frame, idx: int):
        range_images, camera_projections, _, range_image_top_pose = parse_range_image_and_camera_projection(frame)
        points_0, cp_points_0, intensity_0 = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            merge_points=MERGE_POINT_FLAG
        )

        points_1, cp_points_1, intensity_1 = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1,
            merge_points=MERGE_POINT_FLAG
        )

        # waymo TOP lidar fov : -17.92 ~ 2.42
        # waymo TOP lidar channels : 64

        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        top_extrinsic = np.reshape(np.array(calibrations[0].extrinsic.transform), [4, 4])

        total_points = []
        for k, v in points_0.items():
            points = np.concatenate([points_0[k], points_1[k]], axis=0)
            intensity = np.concatenate([intensity_0[k], intensity_1[k]], axis=0)

            # concatenate x,y,z and intensity
            point_cloud = np.column_stack((points, intensity))
            if 'like' not in self.dst_db_type:
                point_cloud = (self.lid_rot @ point_cloud.T).T
            point_cloud[:, 3] = intensity

            if MERGE_POINT_FLAG:
                total_points.append(point_cloud)

            if not MERGE_POINT_FLAG:
                if not os.path.exists(f'{self.dst_dir}lidar/{self.int_to_lid_name[int(k)]}'):
                    os.makedirs(f'{self.dst_dir}lidar/{self.int_to_lid_name[int(k)]}')
                point_cloud.astype(np.float32).tofile(f'{self.dst_dir}lidar/{self.int_to_lid_name[int(k)]}/{idx:06d}.bin')

        if MERGE_POINT_FLAG:
            total_points = np.concatenate(total_points, axis=0)
            total_points[:, :3] -= top_extrinsic[:3, 3].T
            total_points.astype(np.float32).tofile(f'{self.dst_dir}lidar/TOP/{idx:06d}.bin')

    def save_label(self, frame, idx: int):
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        top_extrinsic = np.reshape(np.array(calibrations[0].extrinsic.transform), [4, 4])

        id_to_bbox = dict()
        id_to_name = dict()
        no_label_cam_name = ['FRONT', 'FRONT_RIGHT', 'FRONT_LEFT', 'SIDE_RIGHT', 'SIDE_LEFT']

        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # waymo: bounding box origin is at the center
                bbox = [label.box.center_x - label.box.length / 2, label.box.center_y - label.box.width / 2,
                        label.box.center_x + label.box.length / 2, label.box.center_y + label.box.width / 2]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name

                if name in no_label_cam_name:
                    no_label_cam_name.remove(self.int_to_cam_name[name])

        for obj in frame.laser_labels:
            # calculate bounding box
            bounding_box = None
            name = None
            id = obj.id
            for lidar in ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            if obj.num_lidar_points_in_box < 1:
                continue

            if bounding_box is None or name is None:
                name = '1'
                bounding_box = (0, 0, 0, 0)

            class_name = self.int_to_class_name[obj.type]

            if 'like' in self.dst_db_type:
                cls_lst = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
                class_name = cls_lst[obj.type]

            x_ = obj.box.center_x
            y_ = obj.box.center_y
            z_ = obj.box.center_z
            height = obj.box.height  # up/down
            width = obj.box.width  # left/right
            length = obj.box.length  # front/back
            rot = obj.box.heading

            if 'kitti' in self.dst_db_type:
                if 'like' not in self.dst_db_type:
                    z -= height / 2
                rot -= np.pi / 2

            if self.dst_db_type != 'udacity':
                line = None
                x, y, z, _ = self.lid_rot @ np.array([x_, y_, z_, 1]).T
                x -= top_extrinsic[0, 3]
                y -= top_extrinsic[1, 3]
                z -= top_extrinsic[2, 3]

                if 'kitti' in self.dst_db_type:
                    if 'like' in self.dst_db_type:
                        line = f'{class_name}, -1, 3, -99, ' \
                               f'-1, -1, -1, -1, ' \
                               f'{height:.4f}, {width:.4f}, {length:.4f}, ' \
                               f'{x:.4f}, {y:.4f}, {z:.4f}, {rot:.4f}, ' \
                               f'-1, {obj.id}\n'
                    else:
                        line = f'{class_name}, -1, 3, -99, ' \
                               f'-1, -1, -1, -1, ' \
                               f'{height:.4f}, {width:.4f}, {length:.4f}, ' \
                               f'{x:.4f}, {y:.4f}, {z:.4f}, {rot:.4f}\n'
                else:
                    x, y, z, _ = self.lid_rot @ np.array([x, y, z, 1]).T

                    if self.dst_db_type == 'nuscenes':
                        rot = Rotation.from_euler('xyz', [0, 0, rot])
                        rot_quat = rot.as_quat()
                        line = f'{class_name}, {x:.4f}, {y:.4f}, {z:.4f}, {width:.4f}, {height:.4f}, {length:.4f}, ' \
                               f'{rot_quat[0]:.4f}, {rot_quat[1]:.4f}, {rot_quat[2]:.4f}, {rot_quat[3]:.4f}, ' \
                               f'0, 0, {int(bounding_box[0]):.4f}, {int(bounding_box[1]):.4f}, {int(bounding_box[2]):.4f}, {int(bounding_box[3]):.4f}\n'

                with open(f'{self.dst_dir}label/{self.int_to_lid_name[1]}/{idx:06d}.txt', 'a') as f:
                    f.write(line)

            rot_ = Q(matrix=self.cam_rot_dict[int(name)][:3, :3])
            yaw, _, _ = rot_.yaw_pitch_roll
            rot -= yaw

            line = None
            x, y, z, _ = self.cam_rot_dict[int(name)] @ np.array([x_, y_, z_, 1]).T
            if 'kitti' in self.dst_db_type:
                rot -= np.pi / 2
                y += height / 2
                if z > 0:
                    if 'like' in self.dst_db_type:
                        line = f'{class_name}, -1, 3, -99, ' \
                               f'{int(bounding_box[0]):.4f}, {int(bounding_box[1]):.4f}, ' \
                               f'{int(bounding_box[2]):.4f}, {int(bounding_box[3]):.4f}, ' \
                               f'{height:.4f}, {width:.4f}, {length:.4f}, ' \
                               f'{x:.4f}, {y:.4f}, {z:.4f}, {rot:.4f}, ' \
                               f'-1, {obj.id}\n'
                    else:
                        line = f'{class_name}, -1, 3, -99, ' \
                               f'{int(bounding_box[0]):.4f}, {int(bounding_box[1]):.4f}, ' \
                               f'{int(bounding_box[2]):.4f}, {int(bounding_box[3]):.4f}, ' \
                               f'{height:.4f}, {width:.4f}, {length:.4f}, ' \
                               f'{x:.4f}, {y:.4f}, {z:.4f}, {rot:.4f}\n'
            else:
                if self.dst_db_type == 'nuscenes':
                    if x > 0:
                        rot = Rotation.from_euler('xyz', [0, 0, rot])
                        rot_quat = rot.as_quat()
                        line = f'{class_name}, {x:.4f}, {y:.4f}, {z:.4f}, {width:.4f}, {height:.4f}, {length:.4f}, ' \
                               f'{rot_quat[0]:.4f}, {rot_quat[1]:.4f}, {rot_quat[2]:.4f}, {rot_quat[3]:.4f}, ' \
                               f'0, 0, {int(bounding_box[0]):.4f}, {int(bounding_box[1]):.4f}, {int(bounding_box[2]):.4f}, {int(bounding_box[3]):.4f}\n'
                elif self.dst_db_type == 'udacity' and bounding_box != (0, 0, 0, 0):
                    line = f'{int(bounding_box[0]):.4f}, {int(bounding_box[1]):.4f}, {int(bounding_box[2]):.4f}, {int(bounding_box[3]):.4f}, {class_name}\n'

            if line is not None:
                with open(f'{self.dst_dir}label/{self.int_to_cam_name[int(name)]}/{idx:06d}.txt', 'a') as f:
                    f.write(line)

        for filename in no_label_cam_name:
            with open(f'{self.dst_dir}label/{filename}/{idx:06d}.txt', 'a') as f:
                f.write('')

    def convert(self):
        print(f'Convert waymo to {self.dst_db_type} Dataset.')

        tfrecord_pathnames = sorted(glob(join(self.src_dir, '*.tfrecord')))
        pathname = tfrecord_pathnames
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        cnt = 0

        for index, data in enumerate(tqdm(dataset)):
            frame = W.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            self.calib_convert(frame, index)
            self.save_image(frame, index)
            self.save_lidar(frame, index)
            self.save_label(frame, index)

            cnt += 1

            if cnt == 100:
                break
