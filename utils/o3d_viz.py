import open3d
import numpy as np


def visualization(vis, points: np.ndarray, labels_3d: list = None, labels_cls: list = None):
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))

    vis = draw_box(vis, labels_3d, labels_cls)

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    cam_param = ctr.convert_to_pinhole_camera_parameters()
    cam_param.extrinsic = np.array([[1., 0., 0., 0.],
                                    [0., -1., 0., 0.],
                                    [0., 0., -1., 150.],
                                    [0., 0., 0., 1.]])
    ctr.convert_from_pinhole_camera_parameters(cam_param)


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, cls):
    for i in range(len(gt_boxes)):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        line_set.paint_uniform_color((1, 0, 0))
        vis.add_geometry(line_set)

    return vis
