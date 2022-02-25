import os
from numpy import save
import tensorflow as tf
import numpy as np
import cv2

tf.compat.v1.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset

from PIL import Image
import io

####### REMOVE IMAGE AND LABELS DIRECTORY BEFORE EXECTUING THIS SCRIPT ##############

SAVE_PATH = '/media/robesafe/SSD_SATA/waymo/training_training_0000/'
FILENAME = os.path.join(SAVE_PATH,'tfrecords')

def main():
    files = sorted([os.path.join(FILENAME,f) for f in os.listdir(FILENAME)])
    # print(files)

    if not os.path.isdir(os.path.join(SAVE_PATH,'images')):
        os.mkdir(os.path.join(SAVE_PATH,'images'))
    if not os.path.isdir(os.path.join(SAVE_PATH,'labels')):
        os.mkdir(os.path.join(SAVE_PATH,'labels'))
    train_set = tf.data.TFRecordDataset(files, compression_type='')

    for i, data in enumerate(train_set):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        for  image in frame.images:
            # extract_camera_data(frame,str(i),save_dir=SAVE_PATH,cam_id=1)   ## Only front images (cam_id = 1)
            extract_camera_data(frame,image,frame.camera_labels,save_dir=SAVE_PATH,cam_id=1)

            
def extract_camera_data(frame,camera_image, camera_labels,save_dir,cam_id):
    image_path = os.path.join(save_dir,'images')
    label_path = os.path.join(save_dir,'labels')

    if camera_image.name == cam_id:

        image = Image.open(io.BytesIO(camera_image.image))
        cv_image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        (im_w,im_h) = image.size 

        for camera_labels in frame.camera_labels:
            if camera_labels.name != camera_image.name and camera_labels.name != cam_id:
                continue

            gt_name = os.path.join(label_path,str(len(os.listdir(image_path)))+'.txt')
            gt = open(gt_name,'w+')

            for label in camera_labels.labels:
                
                if label.type == 1:  #Only cars
                    x = label.box.center_x/im_w
                    y = label.box.center_y/im_h
                    h = label.box.length/im_h
                    w = label.box.width/im_w

                    # cv2.circle(cv_image,(int(round(label.box.center_x)),int(round(label.box.center_y))),5,(0,0,255),-1)
                    gt.write(str(0)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h)+'\n') ## Only cars, 0 is stored in labels class
        
        f_name = os.path.join(image_path,str(len(os.listdir(image_path)))+'.jpeg')
        
        n = int(f_name.split('/')[-1].split('.')[0])
        if (n)%100 == 0:
            print(f_name)
        cv2.imwrite(f_name,cv_image)
        gt.close()



if '__main__' == __name__:
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # print(gpu_devices)
    for device in gpu_devices:
       tf.config.experimental.set_memory_growth(device, True)
    main()