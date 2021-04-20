import cv2
import numpy as np
import pandas as pd
import os

start_path = '../pictures/'
end_path = '../processed_test_pictures/'

df = pd.read_csv('../data/dataset.csv', sep=';')

def clahe(filename):
    img = cv2.imread(start_path+filename, 1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_l = clahe.apply(l)
    merged_channels = cv2.merge((clahe_l, a, b))
    clahe = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    cv2.imwrite(end_path+filename, clahe)
    

def clahe_with_gender_division(start_path):
    men_count = 0
    women_count = 0
    women_max_count = df['gender'].value_counts().K
    
    for filename in os.listdir(start_path):
        row = df.loc[df['image'] == filename]
        gender = row['gender'].values[0]
        
        if (men_count >= women_max_count) and (women_count >= women_max_count):
            break
            
        elif gender == 'K':
            women_count+=1
            clahe(filename)
        
        elif gender == 'M' and men_count < women_max_count:
            men_count+=1
            clahe(filename)
clahe_with_gender_division(start_path)
#str(index)+'.jpg'
#clahe(start_path)s