import os
import io
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

from threading import Thread



# Transforms hdf5 files to kitti format

DATASET_PATH = "/media/robesafe/SSD_SATA/shift_dataset/"
# COMPRESSED_PATH = "/home/robesafe/Descargas/"
COMPRESSED_PATH = "/media/robesafe/ff0fec18-b200-4f1b-b17e-f2c93f81163b/shift_dataset/"



def main():
    N_SEQ = 130
    split = 'training'

    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'image_2')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'image_2'))
    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'image_3')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'image_3'))
    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'depth_map')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'depth_map'))
    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'depth_image')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'depth_image'))

    left_im_hdf5 = os.path.join(COMPRESSED_PATH, split, "img_left.hdf5")
    right_im_hdf5 = os.path.join(COMPRESSED_PATH, split,"img.hdf5")
    depth_hdf5 = os.path.join(COMPRESSED_PATH, split,"depth.hdf5")

    with h5py.File(left_im_hdf5, 'r') as f:
        left_nseq = len(list(f.keys()))
    with h5py.File(right_im_hdf5, 'r') as f:
        right_nseq = len(list(f.keys()))
    with h5py.File(depth_hdf5, 'r') as f:
        depth_nseq = len(list(f.keys()))

    assert left_nseq == right_nseq == depth_nseq, "Number of sequences in hdf5 files is not the same"
    assert left_nseq >= N_SEQ, "Number of sequences in hdf5 files is less than the specified number"

    x = Thread(target=convert_left_images,
        args=(left_im_hdf5,0,'training','training.txt',130,))
    y = Thread(target=convert_right_images,
        args=(right_im_hdf5,0,'training',130,))
    z = Thread(target=convert_depths,
        args=(depth_hdf5,0,'training',130,))

    x.start()
    y.start()
    z.start()

    x.join()
    y.join()
    z.join()

    n_training_img = len(list(os.listdir(os.path.join(DATASET_PATH, split, 'image_2'))))
    print("Done training set")

    split = 'validation' ## Kitti format has validation and traning mixed
    N_SEQ = 50

    left_im_hdf5 = os.path.join(COMPRESSED_PATH, split, "img_left.hdf5")
    right_im_hdf5 = os.path.join(COMPRESSED_PATH, split,"img.hdf5")
    depth_hdf5 = os.path.join(COMPRESSED_PATH, split,"depth.hdf5")

    with h5py.File(left_im_hdf5, 'r') as f:
        left_nseq = len(list(f.keys()))
    with h5py.File(right_im_hdf5, 'r') as f:
        right_nseq = len(list(f.keys()))
    with h5py.File(depth_hdf5, 'r') as f:
        depth_nseq = len(list(f.keys()))

    assert left_nseq == right_nseq == depth_nseq, "Number of sequences in hdf5 files is not the same"
    assert left_nseq >= N_SEQ, "Number of sequences in hdf5 files is less than the specified number"

    x = Thread(target=convert_left_images,
        args=(left_im_hdf5,n_training_img,'training','validation.txt',50,))
    y = Thread(target=convert_right_images,
        args=(right_im_hdf5,n_training_img,'training',50,))
    z = Thread(target=convert_depths,
        args=(depth_hdf5,n_training_img,'training',50,))

    x.start()
    y.start()
    z.start()





def convert_left_images(hdf5_file, idx_offest = 0, split = 'training', list_txt = 'training.txt', n_seq = 1):
    
    print(f"Converting left images from {n_seq} sequences...")
    im_idx = idx_offest
    training_txt = open(os.path.join(DATASET_PATH, list_txt), 'w')
    with h5py.File(hdf5_file, 'r') as f:
        seq_names = list(f.keys())[:n_seq]
        for seq in tqdm(seq_names):
            for im in f[seq]:
                data = np.array(f[seq+'/'+im])
                im = Image.open(io.BytesIO(data))
                im.save(os.path.join(DATASET_PATH, split, 'image_2', str(im_idx).zfill(6) + '.png'))
                training_txt.write(str(im_idx).zfill(6) + '\n')
                im_idx += 1


def convert_right_images(hdf5_file, idx_offest = 0, split = 'training', n_seq = 1):

    print(f"Converting right images from {n_seq} sequences...")
    im_idx = idx_offest
    with h5py.File(hdf5_file, 'r') as f:
        seq_names = list(f.keys())[:n_seq]
        for seq in tqdm(seq_names):
            for im in f[seq]:
                data = np.array(f[seq+'/'+im])
                im = Image.open(io.BytesIO(data))
                im.save(os.path.join(DATASET_PATH, split, 'image_3', str(im_idx).zfill(6) + '.png'))
                im_idx += 1


def convert_depths(hdf5_file, idx_offest = 0, split = 'training', n_seq = 1):

    print(f"Converting depths from {n_seq} sequences...")

    im_idx = idx_offest
    with h5py.File(hdf5_file, 'r') as f:
        seq_names = list(f.keys())[:n_seq]
        for seq in tqdm(seq_names):
            for dep in f[seq]:               
                data = np.array(f[seq+'/'+dep])
                img = Image.open(io.BytesIO(data))
            
                depth = np.array(img, dtype=np.float16)

                DEPTH_C = np.array(1000.0 / (256 * 256 * 256 - 1), np.float16)
                depth = (256 * 256 * depth[:, :, 2] +  256 * depth[:, :, 1] + depth[:, :, 0]) * DEPTH_C  # in meters
                np.save(os.path.join(DATASET_PATH, split, 'depth_map', str(im_idx).zfill(6) + '.npy'), depth)

                # Convert to kitti format to visualize
                # depth = depth * 256
                # kitti_form = Image.fromarray(depth.astype('uint16'))
                # kitti_form.save(os.path.join(DATASET_PATH, 'depth_image', str(im_idx).zfill(6) + '.png'))

                im_idx += 1
            


if __name__ == '__main__':
    main()