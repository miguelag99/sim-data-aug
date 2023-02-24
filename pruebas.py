import numpy as np
import os
from PIL import Image
import mayavi.mlab

import sys


def project_velo_to_image(velo, P2, Tr_velo_to_cam):
    velo = velo[:, :3]
    velo = np.hstack((velo, np.ones((velo.shape[0], 1))))
    velo = np.dot(Tr_velo_to_cam, velo.T).T
    velo = velo[:, :3]
    velo = velo[velo[:, 2] > 0]
    depths = velo[:, 2]

    velo = np.hstack((velo, np.ones((velo.shape[0], 1))))
    velo = np.dot(P2, velo.T).T
    velo = (velo[:, :2] / velo[:, 2:]).astype(np.int32)
    return np.hstack((velo, depths.reshape(-1, 1)))



DATASET_PATH = '/media/robesafe/SSD_SATA/shift_dataset/training'
KITTI_PATH = '/media/robesafe/SSD_SATA/KITTI_DATASET/training'

f_path = os.path.join(DATASET_PATH, 'depth_map', '008260.npy')
velo_path = os.path.join(DATASET_PATH, 'velodyne', '008260.bin')
depth_path = os.path.join(DATASET_PATH, 'depth_map', '008260.npy')

##########################
f = np.load(f_path)
file_name = f_path.split('/')[-1].split('.')[0]
        
f[f > 100] = 0
f[f == -1] = 0


# f = f*256.
# f = Image.fromarray(f.astype('uint16'))
# f.show()
# input()


##########################

velo = np.fromfile(velo_path,dtype=np.float32, count=-1).reshape([-1,4])

x = np.asarray(velo[:, 0]).reshape(-1)  # x position of point
y = np.asarray(velo[:, 1]).reshape(-1)  # y position of point
z = np.asarray(velo[:, 2]).reshape(-1)  # z position of point
r = np.asarray(velo[:, 3] ).reshape(-1) # reflectance value of point

fig = mayavi.mlab.figure(bgcolor=(0,0,0),size=(640,500))

mayavi.mlab.points3d(x, -y, z,

                    mode="point",
                    colormap='spectral', # 'bone', 'copper', 'gnuplot'
                    color=(1, 1, 1),   # Used a fixed (r,g,b) instead
                    figure=fig,
                    )  


mayavi.mlab.show()

##########################

P2 = np.array([[640, 0, 640, 0],
                [0, 640, 400, 0],
                [0, 0, 1, 0]])
Tr_velo_to_cam = np.array([[0, 1, 0, 0.2],
                            [0, 0, -1, 0],
                            [1, 0, 0, -0.5]])

# np.set_printoptions(threshold=sys.maxsize)
im_points = project_velo_to_image(velo, P2, Tr_velo_to_cam)
print(im_points)


depth_map = np.zeros((800, 1280)) - 1

for idx in range(im_points.shape[0]):
    u = int(im_points[idx, 0])
    v = int(im_points[idx, 1])
    z = im_points[idx, 2]
    if v < 800 and u > 0 and u < 1280 and v > 0:
        depth_map[v,u] = z

depth_map[depth_map == -1] = 0
depth_map *= 256.
depth_img = Image.fromarray(depth_map.astype('uint16'))

# depth_img.show()
# input()

##########################

depth_gt = np.load(depth_path)
depth_gt[depth_gt == -1] = 0
print(depth_gt.shape)
print(depth_gt)
depth_gt *= 256.
depth_gt = Image.fromarray(depth_gt.astype('uint16'))
depth_gt.show()
