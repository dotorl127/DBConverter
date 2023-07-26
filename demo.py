import os
import argparse
import numpy as np

import mayavi.mlab as mlab
import cv2
import open3d as o3d

from utils import visulize as V
from utils import util


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', help='Directory to load Dataset')
    parser.add_argument('--dataset_type', help='Type Name of Dataset to Visulization')
    parser.add_argument('--vis_type', type=str, default='3d', help='Type of visualization[2d, 3d]')
    args = parser.parse_args()

    assert args.dataset_type in ['kitti', 'waymo', 'nuscenes', 'udacity', 'kitti-like'], \
        f'Invalid Dataset Type Please Check {args.dataset_type}'

    root_path = args.root_path
    if root_path[-1] != '/': root_path += '/'

    camera_names = sorted(os.listdir(f'{root_path}camera/'))

    if args.vis_type == '3d':
        assert args.dataset_type in ['kitti', 'waymo', 'nuscenes', 'kitti-like'], f'Udacity dataset does not support 3D visualize'
        assert os.path.exists(f'{root_path}lidar/'), f'LiDAR point cloud data has not found'

        points_dir_name = None
        lid_lst = os.listdir(f'{root_path}lidar/')
        for lid_name in lid_lst:
            if lid_name in ['velodyne', 'LIDAR_TOP', 'TOP', 'lidar(00)']:
                points_dir_name = lid_name

        filenames = sorted(os.listdir(f'{root_path}lidar/{points_dir_name}'))
        pts_ext = filenames[0][-3:]
        filenames = [filename.rstrip(pts_ext) for filename in filenames]

        for filename in filenames[10:]:
            points = None
            labels_3d = None
            labels_cls = None
            for lid_name in lid_lst:
                if lid_name in ['velodyne', 'LIDAR_TOP', 'TOP', 'lidar(00)']:
                    if pts_ext != 'pcd':
                        points = np.fromfile(f'{root_path}lidar/{points_dir_name}/{filename}{pts_ext}',
                                             dtype=np.float32).reshape(-1, 4)[:, :3]
                    else:
                        pcd = o3d.io.read_point_cloud(f'{root_path}lidar/{points_dir_name}/{filename}{pts_ext}')
                        points = np.asarray(pcd.points)
                    with open(f'{root_path}label/{lid_name}/{filename}txt', 'r') as f:
                        labels_3d = []
                        labels_cls = []
                        lines = f.readlines()
                        for line in lines:
                            label = line.strip().split(', ')
                            # label 3d format is [class_name, x, y, z, width, length, height, rotation_z]
                            _, label_3d, label_cls = util.parse_label(args.dataset_type, label, args.vis_type)
                            labels_3d.append(label_3d)
                            labels_cls.append(label_cls)

            V.visualization(points, labels_3d, labels_cls)
            mlab.show(stop=True)
            mlab.close()
    else:
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

    print('Demo Done')
