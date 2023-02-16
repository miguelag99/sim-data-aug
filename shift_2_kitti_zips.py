import os
import shutil
import numpy as np
from zipfile import ZipFile
from PIL import Image
from tqdm import tqdm
import open3d as o3d

from threading import Thread



# Transforms hdf5 files to kitti format

DATASET_PATH = "/media/robesafe/SSD_SATA/shift_dataset/"
# COMPRESSED_PATH = "/home/robesafe/Descargas/"
COMPRESSED_PATH = "/media/robesafe/ff0fec18-b200-4f1b-b17e-f2c93f81163b/shift_dataset/"



def main():
    n_seq = 110
    split = 'training'

    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'image_2')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'image_2'))
    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'image_3')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'image_3'))
    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'depth_map')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'depth_map'))
    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'depth_image')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'depth_image'))
    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'calib')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'calib'))
    if not os.path.isdir(os.path.join(DATASET_PATH, split, 'velodyne')):
        os.mkdir(os.path.join(DATASET_PATH, split, 'velodyne'))

    left_im_zip = os.path.join(COMPRESSED_PATH, split, "img_left.zip")
    right_im_zip = os.path.join(COMPRESSED_PATH, split,"img.zip")
    depth_zip = os.path.join(COMPRESSED_PATH, split,"depth.zip")
    velo_zip = os.path.join(COMPRESSED_PATH, split,"lidar.zip")

    with ZipFile(left_im_zip, 'r') as zip:
        left_nseq = len(list(zip.namelist()))
    with ZipFile(right_im_zip, 'r') as zip:
        right_nseq = len(list(zip.namelist()))
    with ZipFile(depth_zip, 'r') as zip:
        depth_nseq = len(list(zip.namelist()))

    assert left_nseq == right_nseq == depth_nseq, "Number of sequences in zip files is not the same"
    assert left_nseq >= n_seq, "Number of sequences in zip files is less than the specified number"

    x = Thread(target=convert_left_images,
        args=(left_im_zip,0,'training','training.txt',n_seq,))
    y = Thread(target=convert_right_images,
            args=(right_im_zip,0,'training',n_seq,))
    z = Thread(target=convert_lidar_pcl,
            args=(velo_zip,0,'training',n_seq,))

    x.start()
    y.start()
    z.start()

    x.join()
    y.join()
    z.join()

    n_training_img = len(list(os.listdir(os.path.join(DATASET_PATH, split, 'image_2'))))
    print("Done training set")

    split = 'validation' ## Kitti format has validation and traning mixed
    n_seq = 30

    left_im_zip = os.path.join(COMPRESSED_PATH, split, "img_left.zip")
    right_im_zip = os.path.join(COMPRESSED_PATH, split,"img.zip")
    depth_zip = os.path.join(COMPRESSED_PATH, split,"depth.zip")
    velo_zip = os.path.join(COMPRESSED_PATH, split,"lidar.zip")


    with ZipFile(left_im_zip, 'r') as zip:
        left_nseq = len(list(zip.namelist()))
    with ZipFile(right_im_zip, 'r') as zip:
        right_nseq = len(list(zip.namelist()))
    with ZipFile(depth_zip, 'r') as zip:
        depth_nseq = len(list(zip.namelist()))

    assert left_nseq == right_nseq == depth_nseq, "Number of sequences in zip files is not the same"
    assert left_nseq >= n_seq, "Number of sequences in zip files is less than the specified number"

    x = Thread(target=convert_left_images,
        args=(left_im_zip,n_training_img,'training','validation.txt',n_seq,))
    y = Thread(target=convert_right_images,
            args=(right_im_zip,n_training_img,'training',n_seq,))
    z = Thread(target=convert_lidar_pcl,
            args=(velo_zip,n_training_img,'training',n_seq,)) 

    x.start()
    y.start()
    z.start()

    x.join()
    y.join()
    z.join()

    # Generate calib files

    print('Generating calib files...')

    # Calib files are the same for all images, referenced to the left front camera
    for im_name in tqdm(sorted(os.listdir(os.path.join(DATASET_PATH, 'training', 'image_2')))):
        im_idx = im_name.split('.')[0]
        with open(os.path.join(DATASET_PATH, 'training', 'calib', im_idx + '.txt'), 'w') as f:
            f.write(f"P0: 640 0 640 0 0 640 400 0 0 0 1 0\n")
            f.write(f"P1: 640 0 640 0 0 640 400 0 0 0 1 0\n")
            f.write(f"P2: 640 0 640 0 0 640 400 0 0 0 1 0\n")
            f.write(f"P3: 640 0 640 0 0 640 400 0 0 0 1 0\n")
            f.write(f"R0_rect: 1 0 0 0 1 0 0 0 1\n")
            f.write(f"Tr_velo_to_cam: 0 1 0 0.2 0 0 -1 0 1 0 0 -0.5\n")
           


    # Next, regarding the intrinsic parameters, they are focal_x = focal_y = 640, (center_x, center_y) = (640, 400). 
    # Note that the focal length is computed using focal_x = focal_y = width / (2 * tan(FoV * np.pi / 360.0)), 
    # which is 640 in our case. All RGB cameras share the same intrinsics.




def convert_left_images(zip_file, idx_offest = 0, split = 'training', list_txt = 'training.txt', n_seq = 1):
    
    print(f"Converting left images from {n_seq} sequences...")
    im_idx = idx_offest
    training_txt = open(os.path.join(DATASET_PATH, list_txt), 'w')
    with ZipFile(zip_file, 'r') as f:

        all_image = np.array(f.namelist())
        seq_names = sorted(list(set(map(lambda x: x.split('/')[0],\
                    all_image))))[0:n_seq]

        for seq in tqdm(seq_names):

            filter_seq = lambda x: x.startswith(seq)
            img_filter = list(map(filter_seq, all_image))

            for im in all_image[img_filter]:

                f.extract(im, path = os.path.join(DATASET_PATH, split, 'image_2'))

                im_jpg = Image.open(os.path.join(DATASET_PATH, split, 'image_2', im))
                im_jpg.save(os.path.join(DATASET_PATH, split, 'image_2', str(im_idx).zfill(6) + '.png'))

                os.remove(os.path.join(DATASET_PATH, split, 'image_2', im))

                training_txt.write(str(im_idx).zfill(6) + '\n')
                im_idx += 1

            os.rmdir(os.path.join(DATASET_PATH, split, 'image_2', seq))


def convert_right_images(zip_file, idx_offest = 0, split = 'training', n_seq = 1):
    
    print(f"Converting right images from {n_seq} sequences...")
    im_idx = idx_offest
    with ZipFile(zip_file, 'r') as f:

        all_image = np.array(f.namelist())
        seq_names = sorted(list(set(map(lambda x: x.split('/')[0],\
                    all_image))))[0:n_seq]

        for seq in tqdm(seq_names):

            filter_seq = lambda x: x.startswith(seq)
            img_filter = list(map(filter_seq, all_image))

            for im in all_image[img_filter]:

                f.extract(im, path = os.path.join(DATASET_PATH, split, 'image_3'))

                im_jpg = Image.open(os.path.join(DATASET_PATH, split, 'image_3', im))
                im_jpg.save(os.path.join(DATASET_PATH, split, 'image_3', str(im_idx).zfill(6) + '.png'))
                os.remove(os.path.join(DATASET_PATH, split, 'image_3', im))

                im_idx += 1

            os.rmdir(os.path.join(DATASET_PATH, split, 'image_3', seq))   

def convert_lidar_pcl(zip_file, idx_offest = 0, split = 'training', n_seq = 1):
    
    print(f"Converting lidar pcl from {n_seq} sequences...")
    pcl_idx = idx_offest
    with ZipFile(zip_file, 'r') as f:

        all_lidar_pcl = np.array(f.namelist())
        seq_names = sorted(list(set(map(lambda x: x.split('/')[0],\
                    all_lidar_pcl))))[0:n_seq]
        
        for seq in tqdm(seq_names):

            filter_seq = lambda x: x.startswith(seq)
            pcl_filter = list(map(filter_seq, all_lidar_pcl))

            for pcl in all_lidar_pcl[pcl_filter]:

                f.extract(pcl, path = os.path.join(DATASET_PATH, split, 'velodyne'))
                velo  = o3d.io.read_point_cloud(os.path.join(DATASET_PATH, split, 'velodyne', pcl))
                arr = np.asarray(velo.points)

                arr_flatten = np.zeros(arr.shape[0] * 4, dtype=np.float32)
                arr_flatten[0::4] = arr[:, 0]
                arr_flatten[1::4] = arr[:, 1]
                arr_flatten[2::4] = arr[:, 2]
                # arr_flatten[3::4] = arr[:, 3]
                arr_flatten[3::4] = 0
                arr_flatten.astype('float32').tofile(os.path.join(DATASET_PATH, split,\
                                             'velodyne', str(pcl_idx).zfill(6) + '.bin'))

                os.remove(os.path.join(DATASET_PATH, split, 'velodyne', pcl))

                pcl_idx += 1

            os.rmdir(os.path.join(DATASET_PATH, split, 'velodyne', seq))   


if __name__ == '__main__':
    main()