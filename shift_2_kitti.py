import os
import io
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

from threading import Thread



# Transforms hdf5 files to kitti format

DATASET_PATH = "/media/robesafe/SSD_SATA/shift_dataset/training"
COMPRESSED_PATH = "/home/robesafe/Descargas/"
N_SEQ = 130

def main():

    left_im_hdf5 = os.path.join(COMPRESSED_PATH, "img_left.hdf5")
    right_im_hdf5 = os.path.join(COMPRESSED_PATH, "img.hdf5")
    depth_hdf5 = os.path.join(COMPRESSED_PATH, "depth.hdf5")

    if not os.path.isdir(os.path.join(DATASET_PATH, 'image_2')):
        os.mkdir(os.path.join(DATASET_PATH, 'image_2'))

    if not os.path.isdir(os.path.join(DATASET_PATH, 'image_3')):
        os.mkdir(os.path.join(DATASET_PATH, 'image_3'))

    if not os.path.isdir(os.path.join(DATASET_PATH, 'depth_map')):
        os.mkdir(os.path.join(DATASET_PATH, 'depth_map'))
    

    if not os.path.isdir(os.path.join(DATASET_PATH, 'depth_image')):
        os.mkdir(os.path.join(DATASET_PATH, 'depth_image'))

    x = Thread(target=convert_left_images,
        args=(left_im_hdf5,))
    y = Thread(target=convert_right_images,
        args=(right_im_hdf5,))
    z = Thread(target=convert_depths,
        args=(depth_hdf5,))

    x.start()
    y.start()
    z.start()


def convert_left_images(hdf5_file):
    
    print(f"Converting left images from {N_SEQ} sequences...")
    im_idx = 0
    training_txt = open(os.path.join(DATASET_PATH, 'training.txt'), 'w')
    with h5py.File(hdf5_file, 'r') as f:
        seq_names = list(f.keys())[:N_SEQ]
        for seq in tqdm(seq_names):
            for im in f[seq]:
                data = np.array(f[seq+'/'+im])
                im = Image.open(io.BytesIO(data))
                im.save(os.path.join(DATASET_PATH, 'image_2', str(im_idx).zfill(6) + '.png'))
                training_txt.write(str(im_idx).zfill(6) + '\n')
                im_idx += 1


def convert_right_images(hdf5_file):

    print(f"Converting right images from {N_SEQ} sequences...")
    im_idx = 0
    with h5py.File(hdf5_file, 'r') as f:
        seq_names = list(f.keys())[:N_SEQ]
        for seq in tqdm(seq_names):
            for im in f[seq]:
                data = np.array(f[seq+'/'+im])
                im = Image.open(io.BytesIO(data))
                im.save(os.path.join(DATASET_PATH, 'image_3', str(im_idx).zfill(6) + '.png'))
                im_idx += 1


def convert_depths(hdf5_file):

    print(f"Converting depths from {N_SEQ} sequences...")



    im_idx = 0
    with h5py.File(hdf5_file, 'r') as f:
        seq_names = list(f.keys())[:N_SEQ]
        for seq in tqdm(seq_names):
            for dep in f[seq]:               
                data = np.array(f[seq+'/'+dep])
                img = Image.open(io.BytesIO(data))
            
                depth = np.array(img, dtype=np.float16)

                DEPTH_C = np.array(1000.0 / (256 * 256 * 256 - 1), np.float16)
                depth = (256 * 256 * depth[:, :, 2] +  256 * depth[:, :, 1] + depth[:, :, 0]) * DEPTH_C  # in meters
                np.save(os.path.join(DATASET_PATH, 'depth_map', str(im_idx).zfill(6) + '.npy'), depth)

                # Convert to kitti format to visualize
                # depth = depth * 256
                # kitti_form = Image.fromarray(depth.astype('uint16'))
                # kitti_form.save(os.path.join(DATASET_PATH, 'depth_image', str(im_idx).zfill(6) + '.png'))

                im_idx += 1
            


if __name__ == '__main__':
    main()