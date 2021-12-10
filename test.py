import database as db
import vision_localization as vl
import cv2 as cv
import numpy as np
import os
import sys

if __name__ == '__main__':
    # argv[1]: database
    # argv[2]: dataset

    database = db.load_database(sys.argv[1])

    image_structs_path = os.path.join(sys.argv[2], 'image_structs.mat') 
    image_structs = db.load_image_structs(image_structs_path)

    match_count = 0
    
    for image_info in image_structs:
        image_name = ''.join(image_info[0])
        if image_name not in database:
            image_path = os.path.join(sys.argv[2], 'jpg_rgb', image_name)
            image = cv.imread(image_path)
            
            res = vl.match_database(image, database, numthreads=1)
            if res != None:
                match_count += 1
                print(image_name + ': match found')
            else:
                print(image_name + ': no match found')

    print('%d / %d matches. %.2f%% accuracy' % (match_count, 
                                                len(image_structs),
                                                match_count / len(image_structs)))
    print('Database size: %d' % len(database))

