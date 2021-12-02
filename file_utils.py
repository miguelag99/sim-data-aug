import numpy as np
import pandas as pd
 

def read_params(file_path):

    camera_param = open(file_path,'r')
    matrix = camera_param.readlines()
    camera_param.close()
    matrix = matrix[0].split(',')
    matrix = [float(i) for i in matrix]
        
    p = np.vstack((matrix[0:4],matrix[4:8],matrix[8:12]))
    return p


def read_labels(file_path):

    labels = pd.read_csv(file_path)
    labels['timestamp'] = labels['timestamp'].map(lambda x: round(x,4))
    return labels


def read_timestamps(file_path):

    times_val = open(file_path,'r')
    val = times_val.readlines()
    times_val.close()
    val = [float(i) for i in val]

    return val