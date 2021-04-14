import cv2
import numpy as np
import os

start_path = 'test_pictures/'
end_path = 'processed_test_pictures/'

#images = ['001.jpg','002.jpg','003.jpg','004.jpg','005.jpg']

def clahe(start_path):
    index = 0
    for filename in os.listdir(start_path):
        img = cv2.imread(start_path+filename, 1)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_l = clahe.apply(l)
        merged_channels = cv2.merge((clahe_l, a, b))
        clahe = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
        cv2.imwrite(end_path+filename, clahe)
        index +=1
#clahe(images)
#str(index)+'.jpg'
clahe(start_path)