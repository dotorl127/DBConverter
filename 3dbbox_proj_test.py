import os

import cv2
import numpy as np
from pyquaternion import Quaternion as Q


dir_name = 'FRONT_RIGHT'
dir_names = os.listdir('/home/moon/DATASET/waymo2kitti/camera')
idx = 1

# read img
for dir_name in dir_names:
    img = cv2.imread(f'/home/moon/DATASET/waymo2kitti/camera/{dir_name}/{idx:06d}.png')

    # read calibration
    with open(f'/home/moon/DATASET/waymo2kitti/calib/{dir_name}/{idx:06d}.txt', 'r') as cf:
        datas = cf.readlines()
        intrinsic = np.array(list(map(float, datas[0].split(': ')[1].split(', ')))).reshape(3, 4)

    # read label
    with open(f'/home/moon/DATASET/waymo2kitti/label/{dir_name}/{idx:06d}.txt', 'r') as lf:
        labels = lf.readlines()
    cls = []
    bbox = []
    cuboid = []
    for label in labels:
        name, _, _, _, x1, y1, x2, y2, h, w, l, x, y, z, rot = label.split(', ')
        cls.append(name)
        bbox.append(list(map(float, [x1, y1, x2, y2])))
        cuboid.append(list(map(float, [x, y, z, h, w, l, rot])))

    # convert img label
    af_bbox = []
    for c in cuboid:
        x, y, z, h, w, l, rot = c
        y -= h / 2

        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = h / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = w / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
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

    imsize = 1000
    ratio = 1080 / 1920
    img = cv2.resize(img, (imsize, int(imsize * ratio)))
    cv2.imshow('proj cuboid test', img)
    cv2.waitKey(0)
