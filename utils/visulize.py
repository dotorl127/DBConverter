import numpy as np
import mayavi.mlab as mlab


def draw_multi_grid_range(fig, grid_size=10, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)
    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def create_origin(fig):
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)
    return fig


def draw_points(fig, points: np.ndarray):
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], mode='point',
                  colormap='gnuplot', scale_factor=1, figure=fig)
    return fig


def rotate_points_along_z(points: np.ndarray, angle: np.ndarray):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack([cosa,  sina, zeros,
                           -sina, cosa, zeros,
                           zeros, zeros, ones]).T.reshape(-1, 3, 3)
    points_rot = points @ rot_matrix
    return points_rot


def boxes_to_corners_3d(boxes3d):
    template = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
                         [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]], dtype=np.float) / 2

    corners3d = np.tile(boxes3d[::, 3:6], 8).reshape((-1, 8, 3)) * template
    corners3d = rotate_points_along_z(corners3d, boxes3d[:, 6])
    corners3d += np.tile(boxes3d[::, 0:3], 8).reshape((-1, 8, 3))
    return corners3d


def draw_3d_bbox(fig, corners3d, cls, color=(1, 0, 0), line_width=2, tube_radius=None):
    for n in range(len(corners3d)):
        b = corners3d[n]  # (8, 3)
        mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig


def visualization(points: np.ndarray, labels_3d: list=None, labels_cls: list=None):
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0), engine=None, size=(600, 600))
    fig = draw_multi_grid_range(fig, bv_range=(-60, -60, 60, 60))
    fig = create_origin(fig)
    fig = draw_points(fig, points)

    if labels_3d is not None and labels_cls is not None:
        # TODO: draw 2d bbox into each camera via opencv
        # draw_2d_bbox()
        labels_3d = np.array(labels_3d, dtype=np.float)
        corners3d = boxes_to_corners_3d(labels_3d)
        fig = draw_3d_bbox(fig, corners3d, labels_cls)

    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)

    return fig
