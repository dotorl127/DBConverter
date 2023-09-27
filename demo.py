import os
import argparse
import numpy as np

import mayavi.mlab as mlab
import cv2
import open3d as o3d
from pyquaternion import Quaternion as Q

from utils import visulize as V
from utils import util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--root_path', help='Directory to load Dataset')
    parser.add_argument('-dt', '--dataset_type', help='Type Name of Dataset to Visulization')
    parser.add_argument('-vt', '--vis_type', type=str, default='3d', help='Type of visualization[2d, 3d, project]')
    args = parser.parse_args()

    assert args.dataset_type in ['kitti', 'waymo', 'nuscenes', 'udacity', 'kitti-like'], \
        f'Invalid Dataset Type Please Check {args.dataset_type}'

    root_path = args.root_path
    if root_path[-1] != '/': root_path += '/'

    camera_names = sorted(os.listdir(f'{root_path}camera/'))

    if args.vis_type == '3d':
        assert args.dataset_type in ['kitti', 'waymo', 'nuscenes', 'kitti-like'], \
            f'Udacity dataset does not support 3D visualize'
        assert os.path.exists(f'{root_path}lidar/'), f'LiDAR point cloud data has not found'

        points_dir_name = None
        lid_lst = os.listdir(f'{root_path}lidar/')
        for lid_name in lid_lst:
            if lid_name in ['velodyne', 'LIDAR_TOP', 'TOP', 'lidar(00)']:
                points_dir_name = lid_name

        filenames = sorted(os.listdir(f'{root_path}lidar/{points_dir_name}'))
        pts_ext = filenames[0][-3:]
        filenames = [filename.rstrip(pts_ext) for filename in filenames]

        for filename in filenames:
            points = None
            labels_3d = None
            labels_cls = None
            for lid_name in lid_lst:
                if lid_name in ['velodyne', 'LIDAR_TOP', 'TOP', 'lidar(00)']:
                    # if pts_ext != 'pcd':
                    #     points = np.fromfile(f'{root_path}lidar/{points_dir_name}/{filename}{pts_ext}',
                    #                          dtype=np.float32).reshape(-1, 4)[:, :3]
                    # else:
                    pcd = o3d.io.read_point_cloud(f'/media/SSD/DATASET/kitti-like/nusc2kitti-like/radar/RADAR_FRONT/000000.pcd')
                    points = np.asarray(pcd.points)
                    with open(f'{root_path}label/{lid_name}/{filename}txt', 'r') as f:
                        labels_3d = []
                        labels_cls = []
                        lines = f.readlines()
                        # for line in lines:
                        #     label = line.strip().split(', ')
                        #     # label 3d format is [class_name, x, y, z, width, length, height, rotation_z]
                        #     _, label_3d, label_cls = util.parse_label(args.dataset_type, label, args.vis_type)
                        #     labels_3d.append(label_3d)
                        #     labels_cls.append(label_cls)

            V.visualization(points, labels_3d, labels_cls)
            mlab.show(stop=True)
            mlab.close()
    elif args.vis_type == '2d':
        filenames = sorted(os.listdir(f'{root_path}label/{camera_names[0]}'))
        filenames = [filename.rstrip('.txt') for filename in filenames]
        ext = os.listdir(f'{root_path}camera/{camera_names[0]}')[0].split('.')[-1]
        for filename in filenames:
            for camera_name in camera_names:
                img = cv2.imread(f'{root_path}camera/{camera_name}/{filename}.{ext}')
                width = img.shape[1]
                height = img.shape[0]
                ratio = height / width
                img = cv2.resize(img, (600, int(600 * ratio)))

                if not os.path.exists(f'{root_path}label/{camera_name}/{filename}.txt'):
                    continue

                with open(f'{root_path}label/{camera_name}/{filename}.txt', 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        label = line.strip().split(', ')
                        # label 2d format is [x1, y1, x2, y2]
                        label_2d, _, cls = util.parse_label(args.dataset_type, label, args.vis_type)

                        if label_2d == [0, 0, 0, 0] or cls == 'None':
                            continue

                        x1, y1, x2, y2 = label_2d

                        x1 = int(x1 * img.shape[1] / width)
                        y1 = int(y1 * img.shape[0] / height)
                        x2 = int(x2 * img.shape[1] / width)
                        y2 = int(y2 * img.shape[0] / height)

                        cv2.putText(img, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cv2.imshow(f'{camera_name}', img)
            cv2.waitKey(0)
    elif args.vis_type == 'project':
        for idx in range(len(os.listdir(f'{root_path}label/{camera_names[0]}'))):
            for camera_name in camera_names:
                img = cv2.imread(f'{root_path}camera/{camera_name}/{idx:06d}.png')

                # read calibration
                with open(f'{root_path}calib/{camera_name}/{idx:06d}.txt', 'r') as cf:
                    datas = cf.readlines()
                    intrinsic = np.eye(4)
                    intrinsic[:3, :3] = np.array(list(map(float, datas[0].split(': ')[1].split(', ')))).reshape(3, 3)
                    if np.all(intrinsic.flatten()[:-1] == 0):
                        intrinsic[:3, :4] = np.array(list(map(float, datas[1].split(': ')[1].split(', ')))).reshape(3, 4)

                # read label
                with open(f'{root_path}label/{camera_name}/{idx:06d}.txt', 'r') as lf:
                    labels = lf.readlines()

                cls = []
                bbox = []
                cuboid = []

                for label in labels:
                    name, _, _, _, x1, y1, x2, y2, h, w, l, x, y, z, rot, _, _ = label.split(', ')
                    cls.append(name)
                    bbox.append(list(map(float, [x1, y1, x2, y2])))
                    cuboid.append(list(map(float, [x, y, z, h, w, l, rot])))

                # convert img label
                af_bbox = []

                for c in cuboid:
                    # x: right / y: down / z: forward in kitti label
                    x, y, z, h, w, l, rot = c
                    # KITTI defines the box center as the bottom center of the object
                    y -= h / 2

                    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
                    y_corners = h / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
                    z_corners = w / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
                    corners = np.vstack((x_corners, y_corners, z_corners))

                    # Rotate
                    # y asix is yaw in kitti label
                    rot = Q(axis=(0, 1, 0), angle=rot).rotation_matrix
                    corners = rot @ corners

                    # Translate
                    corners[0, :] = corners[0, :] + x
                    corners[1, :] = corners[1, :] + y
                    corners[2, :] = corners[2, :] + z
                    corners = corners.T

                    # projection to camera
                    n = corners.shape[0]
                    corners_extend = np.hstack((corners, np.ones((n, 1))))
                    pts_2d = np.dot(corners_extend, np.transpose(intrinsic))  # nx3
                    pts_2d[:, 0] /= pts_2d[:, 2]
                    pts_2d[:, 1] /= pts_2d[:, 2]
                    af_bbox.append(pts_2d[:, :2])

                # visualization
                for cls, coor in zip(cls, af_bbox):
                    coor = coor.astype(np.int32)
                    for k in range(0, 4):
                        i, j = k, (k + 1) % 4
                        cv2.line(img, (coor[i, 0], coor[i, 1]), (coor[j, 0], coor[j, 1]), (0, 0, 255), 2)
                        i, j = k + 4, (k + 1) % 4 + 4
                        cv2.line(img, (coor[i, 0], coor[i, 1]), (coor[j, 0], coor[j, 1]), (0, 0, 255), 2)
                        i, j = k, k + 4
                        cv2.line(img, (coor[i, 0], coor[i, 1]), (coor[j, 0], coor[j, 1]), (0, 0, 255), 2)

                width = img.shape[1]
                height = img.shape[0]
                ratio = height / width
                img = cv2.resize(img, (600, int(600 * ratio)))
                cv2.imshow(f'{camera_name} projected', img)
            cv2.waitKey(0)
    else:
        print('should select one of [2d, 3d, project]')
    print('Demo Done')
