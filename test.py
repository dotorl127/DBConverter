import numpy as np
import mayavi.mlab as mlab
from utils import visulize as V


points = np.load('/home/moon/Downloads/anno/000001.npy')
with open('/home/moon/Downloads/anno/000001.txt', 'r') as f:
    labels = f.readlines()

labels_cls = []
labels_3d = []
for label in labels:
    parsed = label.strip().split(', ')

    if parsed[0] != 'DontCare' and parsed[0] != 'Misc':
        labels_cls.append(parsed[0])
        x, y, z = list(map(float, parsed[5:8]))
        l, w, h = list(map(float, parsed[8:11]))
        z += h / 2
        rot = parsed[11]
        labels_3d.append([x, y, z, w, l, h, rot])

V.visualization(points, labels_3d, labels_cls)
mlab.show(stop=True)
mlab.close()
