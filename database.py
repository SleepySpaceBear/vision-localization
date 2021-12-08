import os
import sys
import cv2 as cv
import numpy as np
import scipy.io as sio
import shutil
import json

def find(item, lis, key=None, equality=None):
    """
    Utility function. 
    Finds item in lis using key on items in lis 
    before equality checking with equality.
    If equality is None, uses the == operator.
    If key is None, equality is tested directly on list elements.
    """
    
    if equality is None:
        if key is None:
            for e in lis:
                if item == e:
                    return True
        else:
            for e in lis:
                if item == key(e):
                    return True
    else: 
        if key is None:
            for e in lis:
                if equality(item,e):
                    return True
        else:
            for e in lis:
                if equality(item, key(e)):
                    return True
    return False

def load_image_structs(path):
    """
    Loads the image structure format for the ActiveVision Dataset
    """
    # image struct members are
    # [0] - name
    # [1] - SO matrix
    # [2] - position
    # [3] - viewing direction
   
    image_structs = sio.loadmat(path)
    image_structs = image_structs['image_structs']
    image_structs = image_structs[0]
    
    return image_structs

def init_database_descriptors(image_structs, images_path, printing=False):
    """
    Returns a database dict using the information from image_structs.
    Each image entry is a dictionary with 4 keys:
        'so_mat'   - the orientation matrix associated with the image
        'pos'      - the position at which the image was taken
        'view_dir' - the viewing direction of the camera when the image was taken
        'kp_desc'  - the keypoint descriptors of the image (for matching)
    """
    database = {}

    sift = cv.SIFT_create()

    # move images to database dict and set-up descriptors
    for image in image_structs:
        name = ''.join(image[0]) 
        img = cv.imread(os.path.join(images_path, name))
        desc = sift.detectAndCompute(img, None)[1]
        database[name] = { 
                    'so_mat': image[1].tolist(),
                    'pos': image[2].tolist(),
                    'view_dir': image[3].tolist(),
                    'kp_desc': desc
                }
        if printing:
            print('%s has been initialized in the database' % name) 
    
    if printing:
        print('Database fully initialized!')
    return database


def _match_database_single(database, 
                           point_match_threshold,
                           discrimination_threshold,
                           printing = False):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    for name, params in database.items():
        if 'matches' not in params:
            params['matches'] = {}
        
        for other_name, other_params in database.items():
            # don't match an image with itself 
            if name == other_name:
                continue
            
            if 'matches' not in other_params:
                other_params['matches'] = {}
            
            if name not in other_params['matches']:
                # match keypoints
                matches = flann.knnMatch(
                            params['kp_desc'], 
                            other_params['kp_desc'], 
                            k=2)

                # count good matches
                match_count = 0
                for m,n in matches:
                    if m.distance < n.distance * point_match_threshold:
                        match_count += 1

                    database[name]['matches'][other_name] = match_count
                    database[other_name]['matches'][name] = match_count
    
    # get rid of false matches
    for name, params in database.items():
        del(params['kp_desc']) 
        
        max_matches = 0
        match_list = []
        for _, elem in params['matches'].items():
            if max_matches < elem:
                max_matches = elem

        thresh = max_matches * discrimination_threshold
        # remember our initial match count
        for name, count in params['matches'].items():
            if thresh <= count:
                match_list.append(name)
        
        # if our match count didn't change, we either matched with everything
        # or we didn't match with everything at all

        # let's assume the latter and just say we have no matches
        if len(match_list) == len(params['matches']):
            params['matches'] = []
        else:
            params['matches'] = match_list


def _match_database_multi(database, 
                   point_match_threshold,
                   discrimination_threshold,
                   num_threads,
                   printing = False):

    import queue
    import threading
    from time import sleep

    match_queue = queue.Queue(len(database))
    database_lock = threading.Lock()
    finished = False

    locks = {}

    def match_images_work():
        # set-up matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
         
        while not finished:
            try: 
                work = match_queue.get(timeout=0.05)
                n1 = work[0]
                n2 = work[1]
                des1 = work[2]
                des2 = work[3]

                # match keypoints
                matches = flann.knnMatch(des1, des2, k=2)

                # count good matches
                match_count = 0
                for m,n in matches:
                    if m.distance < n.distance * point_match_threshold:
                        match_count += 1
        
                locks[n1].acquire()
                locks[n2].acquire()
                database[n1]['matches'][n2] = match_count
                database[n2]['matches'][n1] = match_count
                locks[n2].release()
                locks[n1].release()
                match_queue.task_done()
                if printing:
                    print('%s matched with %s!' % (n1, n2))
            except:
                pass

    threads = []
    for i in range(num_threads):
        threads.append(threading.Thread(target=match_images_work)) 
        threads[i].start()

    for name, params in database.items():
        if 'matches' not in params:
            locks[name] = threading.Lock()
            params['matches'] = {}
        
        for other_name, other_params in database.items():
            # don't match an image with itself 
            if name == other_name:
                continue
            
            if 'matches' not in other_params:
                locks[other_name] = threading.Lock()
                other_params['matches'] = {}

            locks[name].acquire()
            locks[other_name].acquire()
            if name not in other_params['matches']:
                params['matches'][other_name] = 0
                other_params['matches'][name] = 0
                
                locks[other_name].release()
                locks[name].release()
                
                match_queue.put((name, other_name,
                        database[name]['kp_desc'],
                        database[other_name]['kp_desc']))
            
            if locks[other_name].locked():
                locks[other_name].release()
            if locks[name].locked():
                locks[name].release()
            
    
    if printing:
        print('Waiting on matching...')
    
    match_queue.join()
    finished = True
    
    # get rid of false matches
    for name, params in database.items():
        del(params['kp_desc']) 
        
        max_matches = 0
        match_list = []
        for _, elem in params['matches'].items():
            if max_matches < elem:
                max_matches = elem

        thresh = max_matches * discrimination_threshold
        # remember our initial match count
        for name, count in params['matches'].items():
            if thresh <= count:
                match_list.append(name)
        
        # if our match count didn't change, we either matched with everything
        # or we didn't match with everything at all

        # let's assume the latter and just say we have no matches
        if len(match_list) == len(params['matches']):
            params['matches'] = []
        else:
            params['matches'] = match_list
    return database

def match_database(database,
                   point_match_threshold,
                   discrimination_threshold,
                   num_threads,
                   printing = False):
    """
    Matches every image in the database with every other image in the 
    database and adds a 'matches' entry to each image's entry in the 
    database. This 'matches' entry is a dict with the keys being the 
    other images in the database and the value being the match count. 
    """

    if num_threads == 1:
        database = _match_database_single(database, 
                point_match_threshold,
                discrimination_threshold,
                printing)
    else:
        database = _match_database_multi(database,
                point_match_threshold,
                discrimination_threshold,
                num_threads,
                printing)
    if printing:
        print("Database matching complete!")
    
    return database

def prune_database(database, printing = False):
    """
    Prunes the database by removing redundant images based on match counts.
    Requires that each image has a 'matches' entry in its dict. The 'matches'
    entry should be a list of images with matches.
    """
    pruned_database = dict()
    represented_images = set()

    image_matches = []
    
    for name, params in database.items():
        image_matches.append(name)

    image_matches.sort(key = lambda a : -len(database[a]['matches']))

    pruned_database[image_matches[0]] = database[image_matches[0]]
    represented_images.update(database[image_matches[0]]['matches'])
    represented_images.update(image_matches[0])

    # add all the images that match only with unrepresented images
    for image in image_matches[1:]:
        if not image in represented_images and not find(True, 
                database[image]['matches'], 
                lambda a : a in represented_images):
            pruned_database[image] = database[image]
            represented_images.update(database[image]['matches'])
            represented_images.update(image)   
    
    
    # add the remaining unmatched images
    for image in image_matches[1:]:
        if not image in represented_images:
            pruned_database[image] = database[image]
            represented_images.update(database[image]['matches'])
            represented_images.update(image)   

    # remove match information since we don't need it
    for name, params in pruned_database.items():
        del(params['matches'])
    
    if printing:
        print('Database pruning complete!')
    return pruned_database

def generate_database(image_structs, 
        images_path, 
        point_match_threshold = 0.7,
        discrimination_threshold = 0.6,
        num_threads = 16,
        printing = False):
    """
    Generates an image matching database with the following parameters.
    """
    
    database = init_database_descriptors(image_structs, images_path, printing)
    
    database = match_database(database, 
                              point_match_threshold, 
                              discrimination_threshold,
                              num_threads,
                              printing)
    
    database = prune_database(database, printing)
    
    return database

def save_database(database, database_path, image_path):
    json_file = open(os.path.join(database_path, 'image_info.json'), 'w')

    # move the image files
    for name in database.keys():
        shutil.copyfile(os.path.join(image_path, name), 
                        os.path.join(database_path, name))
    json.dump(database, json_file)

if __name__ == '__main__':
    # argv[1] = database path
    # argv[2] = active vision instance path

    database_path = sys.argv[1]
    
    dataset_path = sys.argv[2]
    dataset_image_path = os.path.join(dataset_path, "jpg_rgb")
  
    # load image struct
    image_structs_path = os.path.join(dataset_path, "image_structs.mat")
    image_structs = load_image_structs(image_structs_path)

    database = generate_database(image_structs, dataset_image_path, printing=True)
    save_database(database, database_path, dataset_image_path)
