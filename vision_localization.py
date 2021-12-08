import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import os
import scipy

def load_image_params(param_file_path):
    param_file = open(param_file_path)
    return json.load(param_file)

def load_database(path_to_database):
    param_file_path = os.path.join(path_to_database, 'image_info.json')
    database = load_image_params(param_file_path)
    sift = cv.SIFT_create()

    for name, info in database.items():
        image_path = os.path.join(path_to_database, name)
        info['image'] = cv.imread(image_path)
        info['kp'] = sift.detectAndCompute(info['image'], None)
    return database

def _match_database_s(query, database, 
        point_match_threshold = 0.7, 
        discrimination_threshold = 0.6,
        max_matches = 3):
    """
    Returns the feature matches for the database and the query image.
    """
    
    sift = cv.SIFT_create()

    # get keypoints and descriptor for query
    query_kp, query_des = sift.detectAndCompute(query,None)

    db_matches = []
    max_match_count = 0

    # do matches with every image in the database
    for name, info in database.items():
        # get keypoints and descriptor for database images
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(query_des, info['kp'][1],k=2)

        good_matches = []
        qp = []
        dp = []

        for m,n in matches:
            if m.distance < point_match_threshold * n.distance:
                good_matches.append(m)
                qp.append(query_kp[m.queryIdx].pt)
                dp.append(info['kp'][0][m.trainIdx].pt)
        
        if len(good_matches) != 0:
            db_matches.append(
                {'matches': (np.transpose(np.array(qp)), np.transpose(np.array(dp))),
                 'pos': np.array(info['pos']),
                 'dir': np.array(info['view_dir']),
                 'so_mat': np.array(info['so_mat'])})

        l = len(good_matches)

        if max_match_count < l:
            max_match_count = l

    # We will to use discrimination_threshold to determine the maximum 
    # multiplicative (log) distance from the max that we will keep.
    best_match_threshold = max_match_count * discrimination_threshold
    good_matches = []

    for matches in db_matches:
        if best_match_threshold <= matches['matches'][0].shape[1]:
            good_matches.append(matches)

    # If we have more than max_matches, we likely have a false positive match.
    # Let's return nothing to indicate that, well, we found nothing. 
    if len(good_matches) > max_matches:
        return None

    return good_matches

def _match_database_multi(query, database, 
        point_match_threshold = 0.7, 
        discrimination_threshold = 0.6,
        max_matches = 3,
        numthreads = 16):
    
    import threading
    import queue
    
    db_matches = []
    max_match_count = 0

    match_lock = threading.Lock()
    matching_queue = queue.Queue()

    finished = False

    def match_worker():
        # make local sift and matcher
        sift = cv.SIFT_create()
        
        query_kp, query_des = sift.detectAndCompute(query,None)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        while not finished:
            try:
                work = matching_queue.get(timeout=0.01)
                good_matches = []
                qp = []
                dp = []
        
                flann = cv.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(query_des, work['kp'][1],k=2)

                for m,n in matches:
                    if m.distance < point_match_threshold * n.distance:
                        good_matches.append(m)
                        qp.append(query_kp[m.queryIdx].pt)
                        dp.append(info['kp'][0][m.trainIdx].pt)
                
                match_lock.acquire()
                if len(good_matches) != 0:
                    db_matches.append(
                        {'matches': (np.transpose(np.array(qp)), 
                            np.transpose(np.array(dp))),
                         'pos': np.array(info['pos']),
                         'dir': np.array(info['view_dir']),
                         'so_mat': np.array(info['so_mat'])})

                l = len(good_matches)

                if max_match_count < l:
                    max_match_count = l
                match_lock.release()
                matching_queue.task_done()
            except:
                pass


    # do matches with every image in the database
    for name, info in database.items():
        matching_queue.put(info)

    matching_queue.join()
    finished = True

    # We will to use discrimination_threshold to determine the maximum 
    # multiplicative (log) distance from the max that we will keep.
    best_match_threshold = max_match_count * discrimination_threshold
    good_matches = []

    for matches in db_matches:
        if best_match_threshold <= matches['matches'][0].shape[1]:
            good_matches.append(matches)

    # If we have more than max_matches, we likely have a false positive match.
    # Let's return nothing to indicate that, well, we found nothing. 
    if len(good_matches) > max_matches:
        return None

    return good_matches

def match_database(query, database,
        point_match_threshold = 0.7,
        discrimination_threshold = 0.6,
        max_matches = 3,
        numthreads = 1):
    if numthreads == 1:
        return _match_database_s(
                query,
                database,
                point_match_threshold,
                discrimination_threshold,
                max_matches)
    else:
        return _match_database_multi(
                query,
                database,
                point_match_threshold,
                discrimination_threshold,
                max_matches,
                numthreads)


if __name__ == '__main__':
    # argv[1] - query image
    # argv[2] - database dir

    # parse cmdline args
    query_image_path = sys.argv[1]
    database_dir = sys.argv[2]

    # load query image
    query_image = cv.imread(query_image_path)

    # load database
    database = load_database(database_dir)

    # find feature matches
    matches = match_database(query_image, database, numthreads=16)
