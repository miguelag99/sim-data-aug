import os
import numpy as np
import cv2
import pandas as pd
import math

from file_utils import read_params,read_labels,read_timestamps
from print_utils import print_2d_box, print_3d_box, print_center
 

LABEL_PATH = "/home/miguel/simulation-data-aug/perception/groundtruth.csv"
CALIB_PATH = "/home/miguel/simulation-data-aug/perception/camera/intrinsic_matrix.txt"
IMAGE_PATH = "/home/miguel/simulation-data-aug/perception/camera/data/"
TIME_FILE = "/home/miguel/simulation-data-aug/perception/camera/timestamp.txt"
KITTI_PATH = "/home/miguel/simulation-data-aug/kitti_transform/"

def main():

    image_files = sorted(os.listdir(IMAGE_PATH))
    image_files = [IMAGE_PATH + name for name in image_files]

    calib = read_params(CALIB_PATH) 
    obj = read_labels(LABEL_PATH)
    timestamps = read_timestamps(TIME_FILE)
    timestamps = [i for i in timestamps]

    for i in range(len(image_files)):

        im = cv2.imread(image_files[i])
        timestamp = timestamps[i]
        # cv2.putText(im,str(timestamp),(0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
        # print(timestamp)
        # print(image_files[i].split('/')[7].split('.')[0])
        name = image_files[i].split('/')[7].split('.')[0]
        
        ## Get objects in the same frame
        labels = obj[obj['timestamp'] == round(timestamp,4)]
        ## Filter objects out of camera fov
        labels = labels[labels['left'] != -1]
        ## Print gt data 
        print_2d_box(im,labels)
        print_center(im,labels,calib)
        print_3d_box(im,labels,calib)
        # cv2.imshow('image',im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(KITTI_PATH+name+".jpg",im)
   

if __name__ == "__main__":
    main()