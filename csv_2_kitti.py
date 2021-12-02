import os
import numpy as np
import pandas as pd
import math

from file_utils import read_labels,read_timestamps



def main(local_path):

    LABEL_PATH = os.path.join(local_path,"perception/groundtruth.csv")
    IMAGE_PATH = os.path.join(local_path,"perception/camera/data/")
    TIME_FILE = os.path.join(local_path,"perception/camera/timestamp.txt")
    TRANSFORM_PATH = os.path.join(local_path,"kitti_format/")

    image_files = sorted(os.listdir(IMAGE_PATH))
    image_files = [IMAGE_PATH + name for name in image_files]
 
    obj = read_labels(LABEL_PATH)
    timestamps = read_timestamps(TIME_FILE)
    timestamps = [i for i in timestamps]

    for i in range(len(image_files)):

        timestamp = timestamps[i]

        name = image_files[i].split('/')[7].split('.')[0]
        name = name + ".txt"

        file = open(os.path.join(TRANSFORM_PATH,name),'w+')

        ## Get objects in the same frame
        labels = obj[obj['timestamp'] == round(timestamp,4)]
        ## Filter objects out of camera fov
        labels = labels[labels['left'] != -1]

        for i in range(len(labels)):

            cl = labels.iloc[i]['type']
            alpha = labels.iloc[i]['alpha']
            bbox = (labels.iloc[i]['left'],labels.iloc[i]['top'],labels.iloc[i]['right'],labels.iloc[i]['bottom'])
            dim = (labels.iloc[i]['h'],labels.iloc[i]['w'],labels.iloc[i]['l'])
            loc_ego = (labels.iloc[i]['x'],labels.iloc[i]['y'],labels.iloc[i]['z'])
            loc_cam = (-loc_ego[1],-(loc_ego[2]-(1.5/2)),(loc_ego[0]+0.41))
            rot = labels.iloc[i]['rotation_z']

            file.write("{} 0.00 0 {} {} {} {} {} {} {} {} {} {} {} {}".format(cl,alpha,bbox[0],bbox[1],bbox[2],bbox[3],\
                dim[0],dim[1],dim[2],loc_cam[0],loc_cam[1],loc_cam[2],rot))
            file.write('\n')
            


       

   

if __name__ == "__main__":

    os.chdir("..")
    path = os.getcwd()
    if not os.path.isdir(os.path.join(path,"kitti_format")):
        os.mkdir(os.path.join(path,"kitti_format"))

    main(path)