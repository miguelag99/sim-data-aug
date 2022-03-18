import argparse
import os
import time
import pandas as pd
import cv2


SCENES = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
SCENE_TYPE = ['15-deg-left','15-deg-right','30-deg-left','30-deg-right','clone','morning','overcast','sunset']


def create_gt(scene_path): # Save gt in .txt file with coco format (rgb_00000.txt)


    siz_im = (375,1242)

    for condition in SCENE_TYPE:

        d = os.path.join(scene_path,condition,'frames/rgb/')
        gt_file = pd.read_csv(os.path.join(scene_path,condition,'bbox.txt'),sep=' ',header=0)
                
        if not os.path.isdir(os.path.join(d,'labels')):
            ## Create new labels folder
            os.mkdir(os.path.join(d,'labels'))

        ## If camera_0 folder has not been renamed to images
        if not os.path.isdir(os.path.join(d,'images')):
            os.system('mv {}Camera_0 {}images'.format(d,d))

        ## Camera_1 will not be used
        if os.path.isdir(os.path.join(d,'Camera_1')):
            os.system('rm -r {}Camera_1'.format(d))
        gt_file = gt_file[gt_file['cameraID']==0]
        #print(gt_file)
        
        n_images = len(os.listdir(os.path.join(d,'images')))
              
        for i in range(n_images):
            new_file = open(os.path.join(d,'labels','rgb_{:0>5}.txt'.format(i)),'w+')
            labels = gt_file[gt_file['frame']==i]
            if len(labels)>0:
                for j in range(len(labels)):

                    bbox = (labels.iloc[j]['left'],labels.iloc[j]['top'],labels.iloc[j]['right'],labels.iloc[j]['bottom'])
                    
                    w = min((bbox[2] - bbox[0])/siz_im[1],1)
                    h = min((bbox[3] - bbox[1])/siz_im[0],1)
                    
                    center = [min((bbox[2] + bbox[0])/(2*siz_im[1]),1),min((bbox[3] + bbox[1])/(2*siz_im[0]),1)]
                    new_file.write("{} {} {} {} {}".format(0,center[0],center[1],w,h))
                    new_file.write('\n')

            new_file.close()
    return 0

def main(dataset_path):
    a = time.time()
    full_path = [os.path.join(dataset_path,d) for d in SCENES]
    print(full_path)
    list(map(create_gt,full_path))
    b = time.time()
    print('mkdir time: {}'.format(b-a))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dir",help ="Vkitti directory",type =str)
    args = parser.parse_args()

    os.chdir(args.dir)
    main(args.dir)