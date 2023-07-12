import numpy as np
import mayavi.mlab as mlab
from utils import visulize as V


points = np.fromfile('/home/moon/Downloads/anno/000001.bin', dtype=np.float32)
points = points.reshape((-1, 4))[:, :3]

V.visualization(points)
mlab.show(stop=True)
mlab.close()
