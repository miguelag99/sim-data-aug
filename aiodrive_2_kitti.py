import os
import numpy as np
import cv2

LABEL_PATH = "/media/robesafe/DATOS_JYM/AIODrive/label_2/"
CALIB_PATH = "/media/robesafe/DATOS_JYM/AIODrive/calib/"
IMAGE_PATH = "/media/robesafe/DATOS_JYM/AIODrive/image_2/"

def main():

    calib_files = sorted(os.listdir(CALIB_PATH))
    labels_files = sorted(os.listdir(LABEL_PATH))
    image_files = sorted(os.listdir(IMAGE_PATH))
    calib_files = [CALIB_PATH + name for name in calib_files[100:101]]
    labels_files = [LABEL_PATH + name for name in labels_files[100:101]]
    image_files = [IMAGE_PATH + name for name in image_files[100:101]]

    for i in range(len(image_files)):
        calib = read_params(calib_files[i]) 
        obj = read_labels(labels_files[i])
    
        im = cv2.imread(image_files[i])
        print_2d_box(im,obj)
        cv2.imshow('image',im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    


def print_2d_box(im,obj):

    print(len(obj))
    for i in range(len(obj)):
    
        p1 = (obj[i]['bbox'][0],obj[i]['bbox'][1])
        p2 = (obj[i]['bbox'][2],obj[i]['bbox'][3])

        # cv2.line(im,p1,p2,(255,0,0),3)
        # cv2.putText(im, str(obj[i]['pos'][2]), (obj[i]['bbox'][0],obj[i]['bbox'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        cv2.rectangle(im,p1,p2,(255,0,0),3)
        



def read_params(file_path):

    camera_param = open(file_path,'r')
    matrix = camera_param.readlines()
    camera_param.close()
    matrix = (matrix[2].split(':'))[1].split(' ')
    matrix.pop(0)
    matrix[11] = matrix[11].rstrip('\n')
    matrix = [float(i) for i in matrix]
    
    p = np.vstack((matrix[0:4],matrix[4:8],matrix[8:12]))
    return p

def read_labels(file_path):
    
    objects = []
    labels = open(file_path,'r')

    for line in labels:
        line = line.split(' ')
        
        if line[11] != '-1000.00' and float(line[20]) < 50 and abs(float(line[18])) < 30:
            print(line)
            data = {
                'bbox' : (int(float(line[11])),int(float(line[12])),int(float(line[13])),int(float(line[14]))),
                # Get object position AIOdrive
                'pos' : (float(line[18]),float(line[19]),float(line[20]))
                
            }
            objects.append(data)

    return objects



if __name__ == "__main__":
    main()