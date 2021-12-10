import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
import scipy
import database as db

def _match_database_s(query, database, 
        point_match_threshold = 0.7, 
        discrimination_threshold = 0.6,
        max_matches = 5):
    """
    Returns the feature matches for the database and the query image.
    """
    
    orb = cv.ORB_create()
    matcher = db.make_matcher()

    # get keypoints and descriptor for query
    query_kp, query_desc = orb.detectAndCompute(query,None)

    db_matches = []
    max_match_count = 0

    # do matches with every image in the database
    for name, info in database.items():
        matches = db.match_images(query_desc, info['desc'], matcher,
                point_match_threshold)
        qp = []
        dp = []

        for m in matches:
            qp.append(query_kp[m.queryIdx].pt)
            dp.append(info['kp'][m.trainIdx].pt)


        if len(matches) != 0:
            db_matches.append(
                {'matches': (np.transpose(np.array(qp)), np.transpose(np.array(dp))),
                 'pos': np.array(info['pos']),
                 'dir': np.array(info['view_dir']),
                 'so_mat': np.array(info['so_mat'])})

        l = len(matches)

        if max_match_count < l:
            max_match_count = l

    # We will to use discrimination_threshold to determine the maximum 
    # multiplicative (log) distance from the max that we will keep.
    best_match_threshold = max_match_count * discrimination_threshold
    good_matches = []
    
    for matches in db_matches:
        if best_match_threshold <= matches['matches'][0].shape[1]:
            good_matches.append([matches])

    # If we have more than max_matches, we likely have a false positive match.
    # Let's return nothing to indicate that, well, we found nothing. 
    if len(good_matches) > max_matches:
        return None

    return good_matches

def _match_database_multi(query, database, 
        point_match_threshold = 0.7, 
        discrimination_threshold = 0.6,
        max_matches = 5,
        numthreads = 16):
    
    import threading
    import queue
    
    db_matches = []
    max_match_count = 0

    match_lock = threading.Lock()
    matching_queue = queue.Queue()

    finished = False

    def match_worker():
        # make local orb and matcher
        orb = cv.SIFT_create()
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    
        query_kp, query_des = orb.detectAndCompute(query,None)
        

        while not finished:
            try:
                work = matching_queue.get(timeout=0.01)
     
                matches = db.match_images(query_desc, info['desc'], matcher,
                point_match_threshold)
                qp = []
                dp = []

                for m in matches:
                    qp.append(query_kp[m.queryIdx].pt)
                    dp.append(info['kp'][m.trainIdx].pt)
                
                if len(matches) != 0:
                    match_lock.acquire()
                    db_matches.append(
                        {'matches': (np.transpose(np.array(qp)), 
                            np.transpose(np.array(dp))),
                         'pos': np.array(info['pos']),
                         'dir': np.array(info['view_dir']),
                         'so_mat': np.array(info['so_mat'])})
                    match_lock.release()
                matching_queue.task_done()
            except Exception as e:
                print(e)
                pass
    threads = []
    for i in range(num_threads):
        threads.append(threading.Thread(target=match_worker))
        threads[i].start()

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
        max_matches = 5,
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
    database = db.load_database(database_dir)

    # find feature matches
    matches = match_database(query_image, database, max_matches = 5, numthreads=16)
