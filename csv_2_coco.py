import os
import numpy as np
import pandas as pd
import math
import cv2
import argparse

from file_utils import read_labels,read_timestamps

## Transofm custom dataset cdv to coco with 2 classes (0 for car and 1 for pedestrian)

def main(local_path):

    LABEL_PATH = os.path.join(local_path,"groundtruth.csv")
    IMAGE_PATH = os.path.join(local_path,"camera/images/")
    TIME_FILE = os.path.join(local_path,"camera/timestamp.txt")
    TRANSFORM_PATH = os.path.join(local_path,"camera/labels/")

    if not os.path.isdir(TRANSFORM_PATH):
        os.mkdir(TRANSFORM_PATH)

    image_files = sorted(os.listdir(IMAGE_PATH))
    image_files = [IMAGE_PATH + name for name in image_files]

    siz_im = cv2.imread(image_files[0])
    siz_im = siz_im.shape[0:2]
    print(siz_im)
    
    obj = read_labels(LABEL_PATH)
    timestamps = read_timestamps(TIME_FILE)
    timestamps = [i for i in timestamps]

    for i in range(len(image_files)):

        timestamp = timestamps[i]
        
        name = image_files[i].split('/')[-1].split('.')[0]
        name = name + ".txt"
        print(name)

        file = open(os.path.join(TRANSFORM_PATH,name),'w+')

        ## Get objects in the same frame
        labels = obj[obj['timestamp'] == round(timestamp,4)]
        ## Filter objects out of camera fov
        labels = labels[labels['left'] != -1]
        labels = labels[labels['occluded'] != 3]
        labels = labels[labels['occluded'] != 2]
        ## Filter far objects
        labels = labels[labels['x'] < 50]

        for i in range(len(labels)):

            cl = labels.iloc[i]['type']
            if cl == 'Car':
                cl = 0
            elif cl == 'Pedestrian':
                cl = 1
            
            bbox = (labels.iloc[i]['left'],labels.iloc[i]['top'],labels.iloc[i]['right'],labels.iloc[i]['bottom'])
            
            w = min((bbox[2] - bbox[0])/siz_im[0],1)
            h = min((bbox[3] - bbox[1])/siz_im[1],1)
            
            center = [min((bbox[2] + bbox[0])/(2*siz_im[1]),1),min((bbox[3] + bbox[1])/(2*siz_im[0]),1)]
            
            if cl == 0:
                file.write("{} {} {} {} {}".format(cl,center[0],center[1],w,h))
                file.write('\n')


    file.close()
            


       

   

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset_dir",help ="Dataset directory",type =str)
    args = parser.parse_args()

    main(args.dataset_dir)