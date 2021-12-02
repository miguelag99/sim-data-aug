import cv2
import numpy as np

from geometric_utils import create_3d_bbox

def print_2d_box(im,obj):

    for i in range(len(obj)):

        p1 = (obj.iloc[i]['left'],obj.iloc[i]['top'])
        p2 = (obj.iloc[i]['right'],obj.iloc[i]['bottom'])

        # cv2.line(im,p1,p2,(255,0,0),3)
        cv2.rectangle(im,p2,p1,(255,0,0),3)
        

def print_center(im,obj,calib):

    for i in range(len(obj)):

        # point =  np.array([-obj.iloc[i]['y'],-(obj.iloc[i]['z']-(1.64/2)),obj.iloc[i]['x']+0.41,1])
        point =  np.array([-obj.iloc[i]['y'],-(obj.iloc[i]['z']-(1.5/2)),obj.iloc[i]['x']+0.41,1])
        # point =  np.array([-obj.iloc[i]['y'],-(obj.iloc[i]['z']),obj.iloc[i]['x']+0.41,1])
        proj = np.dot(calib,point)
        proj[0] = proj[0]/proj[2]
        proj[1] = proj[1]/proj[2]
        p = (int(proj[0]),int(proj[1]))
        # print(proj)
        cv2.circle(im, p, 2, (255,0,255), thickness=2)


def print_3d_box(im,obj,calib):

    for i in range(len(obj)):

        rot = obj.iloc[i]['rotation_z']
        dim = (obj.iloc[i]['h'],obj.iloc[i]['w'],obj.iloc[i]['l'])
        # loc = (-obj.iloc[i]['y'],-(obj.iloc[i]['z']-((1.64)/2)),obj.iloc[i]['x']+0.41)
        loc = (-obj.iloc[i]['y'],-(obj.iloc[i]['z']-(1.5/2)),obj.iloc[i]['x']+0.41)
        ## Cam altura :1.64 y 0.41
        ## Lidar altura: 1.95 y 0

        
        
        box = create_3d_bbox(rot,dim,loc)

        proj = np.dot(calib,box)

        points = []
        for j in range (np.shape(proj)[1]):
            points.append((int(proj[0,j]/proj[2,j]),int(proj[1,j]/proj[2,j])))
        
        for p in points:
            cv2.circle(im, p, 2, (0,0,255), thickness=2)
        
        cv2.line(im, points[4], points[5], (0,255,0), 1)
        cv2.line(im, points[4], points[7], (0,255,0), 1)
        cv2.line(im, points[6], points[5], (0,255,0), 1)
        cv2.line(im, points[6], points[7], (0,255,0), 1)
    
        cv2.line(im, points[0], points[1], (0,255,0), 1)
        cv2.line(im, points[0], points[3], (0,255,0), 1)
        cv2.line(im, points[2], points[1], (0,255,0), 1)
        cv2.line(im, points[2], points[3], (0,255,0), 1)