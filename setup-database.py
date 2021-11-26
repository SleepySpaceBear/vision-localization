import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import misc
import scipy.io as sio
import shutil

def load_image_structs(path):
    """
    Loads the image structure format for the ActiveVision Dataset
    """
    image_structs = sio.loadmat(path)
    image_structs = image_structs['image_structs']
    image_structs = image_structs[0]
    return image_structs

def get_image_params(name, image_structs):
    """
    Given an image name, return a dict of the image/camera parameters
    """
    image_params = None
    for img in image_structs:
        if name == img[0]:
            image_params = img
            break

    p = np.array(image_params[3]).flatten()
    d = np.array(image_params[4]).flatten()
    d = d / np.linalg.norm(d) # normalize d

    return {'pos': p.tolist(), 'dir': d.tolist() }


if __name__ == '__main__':
    # argv[1] = database path
    # argv[2] = active vision dataset path
    # argv[3] = path to list of database images

    database_path = sys.argv[1]
    
    dataset_path = sys.argv[2]
    dataset_image_path = os.path.join(dataset_path, "jpg_rgb")

    database_image_list = sys.argv[3]
  
    # load image struct
    image_structs_path = os.path.join(dataset_path, "image_structs.mat")
    image_structs = load_image_structs(image_structs_path)

    image_list = open(database_image_list, 'r')

    database_dict_path = os.path.join(database_path, "image_params.json")
    database_dict = {}

    for image_name in image_list.readlines():
        # copy the file into the database
        image_name = image_name[0:-1]
        old_image_path = os.path.join(dataset_image_path, image_name)
        new_image_path = os.path.join(database_path, image_name)
        shutil.copyfile(old_image_path, new_image_path)

        image_params = get_image_params(image_name, image_structs)

        database_dict[image_name] = image_params 
    
    # write database dict to disk
    database_dict_file = open(database_dict_path, 'w')
    json.dump(database_dict, database_dict_file)
