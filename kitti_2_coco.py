import os
import argparse

## Convert Kitti labels to Coco with two classes, Car (0) and Pedestrian (1)

def main(source_path,dest_path):

    kitti_labels = sorted(os.listdir(source_path))
    im_res = (1224,370)
    
    for file in kitti_labels:

        f_kitti = open(os.path.join(source_path,file),'r')
        f_coco = open(os.path.join(dest_path,file),'w+')

        for obj in f_kitti:

            data = obj.split(' ')

            if data[0] == 'Car' or data[0] == 'Pedestrian' or data[0] == 'Cyclist':

                if data[0] == 'Car':
                    obj_class = 0
                elif data[0] == 'Pedestrian':
                    obj_class = 1
                elif data[0] == 'Cyclist':
                    obj_class = 2
                
                data[1:] = [float(n) for n in data[1:]]
                w = min((data[6] - data[4])/im_res[0],1)
                h = min((data[7] - data[5])/im_res[1],1)
                center = [min((data[6] + data[4])/(2*im_res[0]),1),min((data[7] + data[5])/(2*im_res[1]),1)]

                f_coco.write("{} {} {} {} {}".format(obj_class,center[0],center[1],w,h))
                f_coco.write('\n')

            # if data[0] == 'Car':
                
            #     obj_class = 0
                                
            #     data[1:] = [float(n) for n in data[1:]]
            #     w = min((data[6] - data[4])/im_res[0],1)
            #     h = min((data[7] - data[5])/im_res[1],1)
            #     center = [min((data[6] + data[4])/(2*im_res[0]),1),min((data[7] + data[5])/(2*im_res[1]),1)]

            #     f_coco.write("{} {} {} {} {}".format(obj_class,center[0],center[1],w,h))
            #     f_coco.write('\n')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--kitti_dir",help ="Kitti directory",type =str)
    args = parser.parse_args()

    os.chdir(args.kitti_dir)

    path = os.getcwd()
    if not os.path.isdir(os.path.join(path,"labels")):
        os.mkdir(os.path.join(path,"labels"))

    main(source_path=os.path.join(path,"label_2"),dest_path=os.path.join(path,"labels"))
