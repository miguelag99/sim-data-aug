import os
import numpy as np
from zipfile import ZipFile
from PIL import Image
from tqdm import tqdm
import open3d as o3d
import io
import json

from threading import Thread

# Transforms zips files to kitti format
DATASET_PATH = "/home/robesafe/Datasets/shift_dataset/"

COMPRESSED_PATH_TRAIN = "/media/robesafe/ff0fec18-b200-4f1b-b17e-f2c93f81163b1/shift_dataset/zips_training/"
COMPRESSED_PATH_VAL = "/media/robesafe/ff0fec18-b200-4f1b-b17e-f2c93f81163b1/shift_dataset/zips_validation/"



def main():
    n_seq = 110
    split = 'training'

    # Create folders

    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)
    if not os.path.isdir(os.path.join(DATASET_PATH, split)):
        os.mkdir(os.path.join(DATASET_PATH, split))

    split_folders = ['image_2', 'image_3', 'depth_map', 'depth_image', 'depth_map_velo', 'calib', 'velodyne', 'label_2']
    for folder in split_folders:
        if not os.path.isdir(os.path.join(DATASET_PATH, split, folder)):
            os.mkdir(os.path.join(DATASET_PATH, split, folder))

    print("Creating {} set...".format(split))

    left_im_zip = os.path.join(COMPRESSED_PATH_TRAIN, "img_left.zip")
    right_im_zip = os.path.join(COMPRESSED_PATH_TRAIN, "img.zip")
    depth_zip = os.path.join(COMPRESSED_PATH_TRAIN, "depth.zip")
    velo_zip = os.path.join(COMPRESSED_PATH_TRAIN, "lidar.zip")
    labels_3d_json = os.path.join(COMPRESSED_PATH_TRAIN, "det_3d_center.json")
    labels_2d_left_json = os.path.join(COMPRESSED_PATH_TRAIN, "det_2d_left.json")

    with ZipFile(left_im_zip, 'r') as zip:
        left_nseq = len(list(zip.namelist()))
    with ZipFile(right_im_zip, 'r') as zip:
        right_nseq = len(list(zip.namelist()))
    with ZipFile(depth_zip, 'r') as zip:
        depth_nseq = len(list(zip.namelist()))

    assert left_nseq == right_nseq == depth_nseq, "Number of sequences in zip files is not the same"
    assert left_nseq >= n_seq, "Number of sequences in zip files is less than the specified number"

    # Launch threads for each type of data

    v = Thread(target=convert_3d_labels,
            args=(labels_3d_json, labels_2d_left_json, 0, "training", n_seq,))
    w = Thread(target=convert_depths,
            args=(depth_zip,0,'training',n_seq,))
    x = Thread(target=convert_left_images,
            args=(left_im_zip,0,'training','training.txt',n_seq,))
    y = Thread(target=convert_right_images,
            args=(right_im_zip,0,'training',n_seq,))
    z = Thread(target=convert_lidar_pcl,
            args=(velo_zip,0,'training',n_seq,))

    # v.start()
    # w.start()
    # x.start()
    # y.start()
    # z.start()

    # v.join()
    # w.join()
    # x.join()
    # y.join()
    # z.join()


    n_training_img = len(list(os.listdir(os.path.join(DATASET_PATH, split, 'image_2'))))
    print("Done training set")

    split = 'validation' ## Kitti format has validation and traning mixed
    n_seq = 30

    print("Creating {} set...".format(split))

    left_im_zip = os.path.join(COMPRESSED_PATH_VAL, "img_left.zip")
    right_im_zip = os.path.join(COMPRESSED_PATH_VAL,"img.zip")
    depth_zip = os.path.join(COMPRESSED_PATH_VAL,"depth.zip")
    velo_zip = os.path.join(COMPRESSED_PATH_VAL,"lidar.zip")
    labels_3d_json = os.path.join(COMPRESSED_PATH_VAL, "det_3d_center.json")
    labels_2d_left_json = os.path.join(COMPRESSED_PATH_VAL, "det_2d_left.json")


    with ZipFile(left_im_zip, 'r') as zip:
        left_nseq = len(list(zip.namelist()))
    with ZipFile(right_im_zip, 'r') as zip:
        right_nseq = len(list(zip.namelist()))
    with ZipFile(depth_zip, 'r') as zip:
        depth_nseq = len(list(zip.namelist()))

    assert left_nseq == right_nseq == depth_nseq, "Number of sequences in zip files is not the same"
    assert left_nseq >= n_seq, "Number of sequences in zip files is less than the specified number"

    v = Thread(target=convert_3d_labels,
            args=(labels_3d_json,labels_2d_left_json, n_training_img, "training", n_seq,))
    w = Thread(target=convert_depths,
            args=(depth_zip,n_training_img,'training',n_seq,))
    x = Thread(target=convert_left_images,
        args=(left_im_zip,n_training_img,'training','validation.txt',n_seq,))
    y = Thread(target=convert_right_images,
            args=(right_im_zip,n_training_img,'training',n_seq,))
    z = Thread(target=convert_lidar_pcl,
            args=(velo_zip,n_training_img,'training',n_seq,)) 
    
    # v.start()
    # w.start()
    # x.start()
    # y.start()
    # z.start()

    # v.join()
    # w.join()
    # x.join()
    # y.join()
    # z.join()

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
            f.write(f"Tr_velo_to_cam: 0 -1 0 0.2 0 0 -1 0 1 0 0 -0.5\n")    # Modified to invert the original y axis
            f.write(f"Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n")


    # Next, regarding the intrinsic parameters, they are focal_x = focal_y = 640, (center_x, center_y) = (640, 400). 
    # Note that the focal length is computed using focal_x = focal_y = width / (2 * tan(FoV * np.pi / 360.0)), 
    # which is 640 in our case. All RGB cameras share the same intrinsics.

    # The tr_velo_to_cam matrix should be Tr_velo_to_cam: 0 1 0 0.2 0 0 -1 0 1 0 0 -0.5 if the original shift coordinates are used.
    # For mmdetection 3d training it is necessary to invert the y axis, so the matrix should be Tr_velo_to_cam: 0 -1 0 0.2 0 0 -1 0 1 0 0 -0.5




def convert_left_images(zip_file, idx_offest = 0, split = 'training', list_txt = 'training.txt', n_seq = 1):
    
    
    im_idx = idx_offest
    training_txt = open(os.path.join(DATASET_PATH, list_txt), 'w')
    with ZipFile(zip_file, 'r') as f:

        all_image = np.array(f.namelist())
        seq_names = sorted(list(set(map(lambda x: x.split('/')[0],\
                    all_image))))[0:n_seq]

        for seq in tqdm(seq_names, desc = 'Converting left images'):
            
            filter_seq = lambda x: x.startswith(seq)
            img_filter = list(map(filter_seq, all_image))

            for im in all_image[img_filter]:

                if '500' in im:     # Last sensor file is not labeled
                    continue

                f.extract(im, path = os.path.join(DATASET_PATH, split, 'image_2'))

                im_jpg = Image.open(os.path.join(DATASET_PATH, split, 'image_2', im))
                im_jpg.save(os.path.join(DATASET_PATH, split, 'image_2', str(im_idx).zfill(6) + '.png'))

                os.remove(os.path.join(DATASET_PATH, split, 'image_2', im))

                training_txt.write(str(im_idx).zfill(6) + '\n')
                im_idx += 1

            os.rmdir(os.path.join(DATASET_PATH, split, 'image_2', seq))


def convert_right_images(zip_file, idx_offest = 0, split = 'training', n_seq = 1):
    
    im_idx = idx_offest
    with ZipFile(zip_file, 'r') as f:

        all_image = np.array(f.namelist())
        seq_names = sorted(list(set(map(lambda x: x.split('/')[0],\
                    all_image))))[0:n_seq]

        for seq in tqdm(seq_names, desc = 'Converting right images'):

            filter_seq = lambda x: x.startswith(seq)
            img_filter = list(map(filter_seq, all_image))

            for im in all_image[img_filter]:

                if '500' in im:     # Last sensor file is not labeled
                    continue

                f.extract(im, path = os.path.join(DATASET_PATH, split, 'image_3'))

                im_jpg = Image.open(os.path.join(DATASET_PATH, split, 'image_3', im))
                im_jpg.save(os.path.join(DATASET_PATH, split, 'image_3', str(im_idx).zfill(6) + '.png'))
                os.remove(os.path.join(DATASET_PATH, split, 'image_3', im))

                im_idx += 1

            os.rmdir(os.path.join(DATASET_PATH, split, 'image_3', seq))   

def convert_lidar_pcl(zip_file, idx_offest = 0, split = 'training', n_seq = 1):
    
    pcl_idx = idx_offest
    with ZipFile(zip_file, 'r') as f:

        all_lidar_pcl = np.array(f.namelist())
        seq_names = sorted(list(set(map(lambda x: x.split('/')[0],\
                    all_lidar_pcl))))[0:n_seq]
        
        for seq in tqdm(seq_names, desc = 'Converting lidar pcl'):

            filter_seq = lambda x: x.startswith(seq)
            pcl_filter = list(map(filter_seq, all_lidar_pcl))

            for pcl in all_lidar_pcl[pcl_filter]:

                if '500' in pcl:    # Last sensor file is not labeled
                    continue

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


def convert_depths(zip_file, idx_offest = 0, split = 'training', n_seq = 1, generate_img = False):

    im_idx = idx_offest
    with ZipFile(zip_file, 'r') as f:
        all_depth = np.array(f.namelist())
        seq_names = sorted(list(set(map(lambda x: x.split('/')[0],\
                    all_depth))))[0:n_seq]
        
        for seq in tqdm(seq_names, desc = 'Converting depths'):

            filter_seq = lambda x: x.startswith(seq)
            depth_filter = list(map(filter_seq, all_depth))

            for dep in all_depth[depth_filter]:     

                if '500' in dep:    # Last sensor file is not labeled
                    continue

                f.extract(dep, path = os.path.join(DATASET_PATH, split, 'depth_map'))
            
                img = Image.open(os.path.join(DATASET_PATH, split, 'depth_map',dep))
            
                depth = np.array(img, dtype=np.float16)

                DEPTH_C = np.array(1000.0 / (256 * 256 * 256 - 1), np.float16)
                depth = (256 * 256 * depth[:, :, 2] +  256 * depth[:, :, 1] + depth[:, :, 0]) * DEPTH_C  # in meters
                # depth[depth>80] = 0
                np.save(os.path.join(DATASET_PATH, split, 'depth_map', str(im_idx).zfill(6) + '.npy'), depth)

                if generate_img:
                    # Convert to kitti format to visualize ()
                    depth = depth * 256
                    kitti_form = Image.fromarray(depth.astype('uint16'))
                    kitti_form.save(os.path.join(DATASET_PATH, split, 'depth_image', str(im_idx).zfill(6) + '.png'))

                os.remove(os.path.join(DATASET_PATH, split, 'depth_map', dep))


                im_idx += 1

            os.rmdir(os.path.join(DATASET_PATH, split, 'depth_map', seq))  
        

def convert_3d_labels(json_3dfile,json_2d_file, idx_offest = 0, split = 'training', n_seq = 1):

    # Extract 3D info from gt file. It does not contain 2D info.
    # All of the objects are in center coordinate frame.

    with open(json_3dfile) as f:
        data_3d = json.load(f)
    
    with open(json_2d_file) as f:
        data_2d = json.load(f)

    lbl_idx = idx_offest

    video_names = set()
    for frame in data_3d["frames"]:
        video_names.add(frame["videoName"])
    seq_videos_3d = sorted(list(video_names))[:n_seq]

    video_names = set()
    for frame in data_2d["frames"]:
        video_names.add(frame["videoName"])
    seq_videos_2d = sorted(list(video_names))[:n_seq]

    assert seq_videos_3d == seq_videos_2d , "3D and 2D sequences are not the same"

    new_data_3d = []
    for frame in data_3d["frames"]:
        if frame["videoName"] in seq_videos_3d:
            new_data_3d.append(frame)
    
    new_data_2d = []
    for frame in data_2d["frames"]:
        if frame["videoName"] in seq_videos_2d:
            new_data_2d.append(frame)

    id_extractor = lambda x: x['id'] if (x["category"] in ["car", "pedestrian", "bicycle"]) else None


    for i in tqdm(range(len(new_data_3d)), desc = 'Converting 3d labels'):
        obj_id_2d = list(map(id_extractor, new_data_2d[i]["labels"]))
        with open(DATASET_PATH + '/' +split + "/label_2/" + str(lbl_idx).zfill(6) + ".txt", "w+") as file:
            for label3d in new_data_3d[i]["labels"]:
                if label3d["category"] in ["car", "pedestrian", "bicycle"]:   # Filter by class.
                    if np.linalg.norm(label3d["box3d"]["location"]) < 60.0:   # Filter by distance.
                        cl = label3d["category"].capitalize()

                        if cl == "Bicycle":
                            cl = "Cyclist"

                        alpha = label3d["box3d"]["alpha"]
                        # Check if the object is in the 2D image and get the bbox
                        # If not, set bbox to 0

                        if label3d["id"] in obj_id_2d:
                            cam_label = new_data_2d[i]['labels'][obj_id_2d.index(label3d["id"])]
                            bbox = (int(cam_label["box2d"]["x1"]), int(cam_label["box2d"]["y1"]),\
                                        int(cam_label["box2d"]["x2"]), int(cam_label["box2d"]["y2"]))
                        else:
                            bbox = (0, 0, 0, 0)

                        dim = label3d["box3d"]["dimension"]
                        rot = label3d["box3d"]["orientation"][2]

                        loc = project_velo_to_image_2_frame(np.array(label3d["box3d"]["location"]))      # Transform center frame to camera front left frame
                        loc[1] += dim[2]/2      # Center of the object is at the bottom of the box



                        if rot > -np.pi/2 and rot < np.pi/2:
                            rot += np.pi/2
                        elif rot < -np.pi/2:
                            rot += np.pi/2
                        else:
                            rot -= 3*np.pi/2

                        if np.sqrt([np.power(loc[0],2) + np.power(loc[2],2)]) > 1:
                            file.write("{} 0.00 0 {} {} {} {} {} {} {} {} {} {} {} {}".format(cl,alpha,bbox[0],bbox[1],bbox[2],bbox[3],\
                                dim[2],dim[1],dim[0],loc[0],loc[1],loc[2],rot))
                            file.write('\n')

        lbl_idx += 1

def project_velo_to_image_2_frame(velo):

    Tr_velo_to_cam = np.array([[0, 1, 0, 0.2],
                                [0, 0, -1, 0],
                                [1, 0, 0, -0.5]])

    velo = np.hstack((velo,  1))
    velo = np.dot(Tr_velo_to_cam, velo.T).T

    return velo

if __name__ == '__main__':
    main()
