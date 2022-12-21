import os

from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs

from typing import List, Tuple, Union
from nuscenes.utils.geometry_utils import view_points

import numpy as np

from dictionary.class_dictionary import nuscenes_dict
from dictionary.rotation_dictionary import cam_rot, lid_rot
from utils.util import check_valid_mat


class nuscenes:
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
        self.cam_rot = cam_rot['nuscenes'][dst_db_type]
        self.cam_rot = check_valid_mat(self.cam_rot)
        self.lid_rot = lid_rot['nuscenes'][dst_db_type]
        self.lid_rot = check_valid_mat(self.lid_rot)
        self.rt_mat_dict = {}
        self.nusc = NuScenes(dataroot=self.src_dir)
        self.cam_name = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                         'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        self.lidar_name = ['LIDAR_TOP']
        self.radar_name = ['RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
                           'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
        self.split_logs = None
        print(f'Set Destination Dataset Type {self.dst_db_type}')

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples

    @staticmethod
    def project_kitti_box_to_image(box: Box, p_left: np.ndarray, imsize: Tuple[int, int]) \
            -> Union[None, Tuple[int, int, int, int]]:
        # Create a new box.
        box = box.copy()

        # KITTI defines the box center as the bottom center of the object.
        # We use the true center, so we need to adjust half height in negative y direction.
        box.translate(np.array([0, -box.wlh[2] / 2, 0]))

        corners = np.array([corner for corner in box.corners().T if corner[2] > 0]).T
        if len(corners) == 0:
            return (0, 0, 0, 0)

        # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
        imcorners = view_points(corners, p_left, normalize=True)[:2]
        bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))

        # Crop bbox to prevent it extending outside image.
        bbox_crop = tuple(max(0, b) for b in bbox)
        bbox_crop = (min(imsize[0], bbox_crop[0]),
                     min(imsize[0], bbox_crop[1]),
                     min(imsize[0], bbox_crop[2]),
                     min(imsize[1], bbox_crop[3]))

        # Detect if a cropped box is empty.
        if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
            bbox_crop = (0, 0, 0, 0)

        return bbox_crop

    @staticmethod
    def box_to_string(name: str,
                      box: Box,
                      bbox_2d: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0),
                      truncation: float = -1.0,
                      occlusion: int = -1,
                      alpha: float = -10.0) -> str:
        # Convert quaternion to yaw angle.
        v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[2], v[0])

        # Prepare output.
        trunc = '{:.2f}'.format(truncation)
        occ = '{:d}'.format(occlusion)
        a = '{:.2f}'.format(alpha)
        bb = '{}, {}, {}, {}'.format(int(bbox_2d[0]), int(bbox_2d[1]), int(bbox_2d[2]), int(bbox_2d[3]))
        hwl = '{:.2}, {:.2f}, {:.2f}'.format(box.wlh[2], box.wlh[0], box.wlh[1])  # height, width, length.
        xyz = '{:.2f}, {:.2f}, {:.2f}'.format(box.center[0], box.center[1], box.center[2])  # x, y, z.
        y = '{:.2f}'.format(yaw)  # Yaw angle.

        output = f'{name}, {trunc}, {occ}, {a}, {bb}, {hwl}, {xyz}, {y}'

        return output

    def convert(self):
        print(f'Convert nuscenes to {self.dst_db_type} Dataset.')
        self.split_logs = create_splits_logs('mini_train', self.nusc)

        idx = 0

        sample_tokens = self._split_to_samples(self.split_logs)
        sample_tokens = sample_tokens[:100]

        r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.
        imsize = (1600, 900)

        for sample_token in sample_tokens:
            print(f'Converting data {idx:06d}...')
            # Get sample data.
            sample = self.nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']

            lidar_token = sample['data']['LIDAR_TOP']
            sd_record_lid = self.nusc.get('sample_data', lidar_token)
            self.calib_dict['LIDAR_TOP'] = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

            for rad_name in self.radar_name:
                rad_token = sample['data'][rad_name]
                sd_record_cam = self.nusc.get('sample_data', rad_token)
                self.calib_dict[rad_name] = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
                filename_rad_full = sd_record_cam['filename']

                src_rad_path = os.path.join(self.nusc.dataroot, filename_rad_full)
                dst_rad_path = f'{self.dst_dir}radar/{rad_name}/{idx:06d}.pcd'
                if not os.path.exists(dst_rad_path):
                    assert not dst_rad_path.endswith('.pcd.bin')
                    pcl = RadarPointCloud.from_file(src_rad_path)
                    with open(dst_rad_path, "w") as rad_file:
                        pcl.points.T.tofile(rad_file)

                with open(f'{self.dst_dir}calib/{rad_name}/{idx:06d}.txt', 'w') as f:
                    line = ', '.join(map(str, transform_matrix(self.calib_dict[rad_name]['translation'],
                                                               Quaternion(self.calib_dict[rad_name]['rotation']),
                                                               inverse=False).reshape(-1).tolist())) + '\n'
                    f.write(f'{rad_name}: {line}')

            for cam_name in self.cam_name:
                cam_token = sample['data'][cam_name]
                sd_record_cam = self.nusc.get('sample_data', cam_token)
                self.calib_dict[cam_name] = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])

                # Combine transformations and convert to KITTI format.
                # Note: cam uses same conventions in KITTI and nuScenes.
                lid_to_ego = transform_matrix(self.calib_dict['LIDAR_TOP']['translation'],
                                              Quaternion(self.calib_dict['LIDAR_TOP']['rotation']),
                                              inverse=False)
                ego_to_cam = transform_matrix(self.calib_dict[cam_name]['translation'],
                                              Quaternion(self.calib_dict[cam_name]['rotation']),
                                              inverse=True)
                velo_to_cam = ego_to_cam @ lid_to_ego
                velo_to_cam_kitti = velo_to_cam @ np.linalg.inv(self.lid_rot)

                # Currently not used.
                imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.

                # Projection matrix.
                cam_intrinsic = np.zeros((3, 4))
                cam_intrinsic[:3, :3] = self.calib_dict[cam_name]['camera_intrinsic']  # Cameras are always rectified.

                # Create KITTI style transforms.
                velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
                velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

                if cam_name == 'CAM_FRONT':
                    # Check that the rotation has the same format as in KITTI.
                    assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
                    assert (velo_to_cam_trans[1:3] < 0).all()

                self.rt_mat_dict[cam_name] = velo_to_cam_kitti

                # Retrieve the token from the lidar.
                # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
                # not the camera.
                filename_cam_full = sd_record_cam['filename']
                # token = '%06d' % token_idx # Alternative to use KITTI names.

                # Convert image (jpg to png).
                src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
                dst_im_path = f'{self.dst_dir}camera/{cam_name}/{idx:06d}.png'
                if not os.path.exists(dst_im_path):
                    im = Image.open(src_im_path)
                    im.save(dst_im_path, "PNG")

                if self.dst_db_type == 'kitti':
                    # Create calibration file.
                    kitti_transforms = dict()
                    kitti_transforms['P2'] = cam_intrinsic  # Left camera transform.
                    kitti_transforms['R0_rect'] = np.eye(3)  # Cameras are already rectified.
                    kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
                    kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti

                    calib_path = f'{self.dst_dir}calib/{cam_name}/{idx:06d}.txt'
                    with open(calib_path, "w") as calib_file:
                        for (key, val) in kitti_transforms.items():
                            val = val.flatten()
                            val_str = '%.12e' % val[0]
                            for v in val[1:]:
                                val_str += ', %.12e' % v
                            calib_file.write('%s: %s\n' % (key, val_str))
                else:
                    calib_path = f'{self.dst_dir}calib/{cam_name}/{idx:06d}.txt'
                    with open(calib_path, "w") as calib_file:
                        mat = np.eye(4)
                        mat[:3, :3] = self.calib_dict[cam_name]['camera_intrinsic']
                        line = ', '.join(map(str, mat.reshape(-1).tolist())) + '\n'
                        calib_file.write(f'{cam_name}_intrinsic: {line}')
                        line = ', '.join(map(str, np.linalg.inv(ego_to_cam).reshape(-1).tolist())) + '\n'
                        calib_file.write(f'{cam_name}_extrinsic: {line}')
                        calib_file.write(f'{cam_name}_width: 1600\n')
                        calib_file.write(f'{cam_name}_height: 900\n')
                        calib_file.write(f'{cam_name}_rolling_shutter_direction: -1\n')
                        calib_file.write(f'LIDAR_TOP_beam_inclinations: -1\n')
                        calib_file.write(f'LIDAR_TOP_beam_inclination_min: -1\n')
                        calib_file.write(f'LIDAR_TOP_beam_inclination_max: -1\n')
                        line = ', '.join(map(str, lid_to_ego.reshape(-1).tolist())) + '\n'
                        calib_file.write(f'LIDAR_TOP_extrinsic: {line}')

            for cam_name in self.cam_name:
                # Write label file.
                label_path = f'{self.dst_dir}label/{cam_name}/{idx:06d}.txt'
                with open(label_path, "w") as label_file:
                    for sample_annotation_token in sample_annotation_tokens:
                        sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

                        # Convert nuScenes category to nuScenes detection challenge category.
                        if sample_annotation['category_name'] in nuscenes_dict[f'to_{self.dst_db_type}']:
                            detection_name = nuscenes_dict[f'to_{self.dst_db_type}'][sample_annotation['category_name']]
                        else:
                            continue

                        # Get box in LIDAR frame.
                        _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token,
                                                                         box_vis_level=BoxVisibility.NONE,
                                                                         selected_anntokens=[sample_annotation_token])
                        box_lidar_nusc = box_lidar_nusc[0]

                        if self.dst_db_type == 'kitti':
                            truncated = 0.0
                            occluded = 0
                            if detection_name is None:
                                continue

                            box_cam_kitti = \
                                KittiDB.box_nuscenes_to_kitti(box_lidar_nusc,
                                                              velo_to_cam_rot=
                                                              Quaternion(matrix=self.rt_mat_dict[cam_name][:3, :3]),
                                                              velo_to_cam_trans=
                                                              self.rt_mat_dict[cam_name][:3, 3], r0_rect=r0_rect)

                            cam_intrinsic = np.zeros((3, 4))
                            cam_intrinsic[:3, :3] = self.calib_dict[cam_name]['camera_intrinsic']

                            bbox_2d = self.project_kitti_box_to_image(box_cam_kitti, cam_intrinsic, imsize=imsize)

                            if bbox_2d == (0, 0, 0, 0) and cam_name != 'CAM_FRONT':
                                continue

                            box_cam_kitti = \
                                KittiDB.box_nuscenes_to_kitti(box_lidar_nusc,
                                                              velo_to_cam_rot=
                                                              Quaternion(matrix=self.rt_mat_dict['CAM_FRONT'][:3, :3]),
                                                              velo_to_cam_trans=
                                                              np.zeros(3), r0_rect=r0_rect)

                            output = self.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                                        truncation=truncated, occlusion=occluded)
                        else:
                            box_cam_kitti = \
                                KittiDB.box_nuscenes_to_kitti(box_lidar_nusc,
                                                              velo_to_cam_rot=
                                                              Quaternion(matrix=self.rt_mat_dict[cam_name][:3, :3]),
                                                              velo_to_cam_trans=
                                                              self.rt_mat_dict[cam_name][:3, 3], r0_rect=r0_rect)

                            cam_intrinsic = np.zeros((3, 4))
                            cam_intrinsic[:3, :3] = self.calib_dict[cam_name]['camera_intrinsic']

                            # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                            bbox_2d = self.project_kitti_box_to_image(box_cam_kitti, cam_intrinsic, imsize=imsize)

                            if bbox_2d == (0, 0, 0, 0) and cam_name != 'CAM_FRONT':
                                continue

                            box_lidar_nusc.rotate(Quaternion(matrix=self.lid_rot[:3, :3]))

                            v = box_lidar_nusc.rotation_matrix @ np.array([1, 0, 0])
                            yaw = -np.arctan2(v[2], v[0])
                            yaw -= np.pi / 2

                            w = int(bbox_2d[2] - bbox_2d[0])
                            h = int(bbox_2d[3] - bbox_2d[1])
                            cx = int(bbox_2d[0] + (w // 2))
                            cy = int(bbox_2d[1] + (h // 2))

                            if self.dst_db_type == 'waymo':
                                output = f'{cx}, {cy}, {w}, {h}, 0, 0, {detection_name}, -1, ' \
                                         f'{box_lidar_nusc.center[0]}, {box_lidar_nusc.center[1]}, {box_lidar_nusc.center[2]}, ' \
                                         f'{box_lidar_nusc.wlh[0]}, {box_lidar_nusc.wlh[1]}, {box_lidar_nusc.wlh[2]}, {yaw}'
                            elif self.dst_db_type == 'udacity':
                                x1, y1, x2, y2 = map(int, bbox_2d)
                                output = f'{x1}, {y1}, {x2}, {y2}, {detection_name}'

                        # Write to disk.
                        label_file.write(output + '\n')

            filename_lid_full = sd_record_lid['filename']

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
            dst_lid_path = f'{self.dst_dir}lidar/LIDAR_TOP/{idx:06d}.bin'
            assert not dst_lid_path.endswith('.pcd.bin')
            pcl = LidarPointCloud.from_file(src_lid_path)
            pcl.rotate(self.lid_rot[:3, :3])  # In KITTI lidar frame.
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            idx += 1
