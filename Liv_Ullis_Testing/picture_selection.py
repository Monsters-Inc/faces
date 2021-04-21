import cv2
import numpy as np
import pandas as pd
import os

start_path = '../pictures/'
end_path_gender = 'processed_gender_pictures/'
end_path_age_and_gender = 'processed_age_and_gender_pictures/'
end_path_age = 'processed_age_pictures/'

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

def write_image_gender(filename):
    img = cv2.imread(start_path+filename, 1)
    cv2.imwrite(end_path_gender+filename, img)

def write_image_age_and_gender(filename):
    img = cv2.imread(start_path+filename, 1)
    cv2.imwrite(end_path_age_and_gender+filename, img)

def write_image_age(filename):
    img = cv2.imread(start_path+filename, 1)
    cv2.imwrite(end_path_age+filename, img)

def gender_division(start_path):
    for f in os.listdir(end_path_age_and_gender):
        os.remove(f)
    
    men_count = 0
    women_count = 0
    women_max_count = df['gender'].value_counts().K
    for filename in os.listdir(start_path):
        row = df.loc[df['image'] == filename]
        gender = row['gender'].values[0]

        if (men_count >= women_max_count) and (women_count >= women_max_count):
            break
            
        elif gender == 'K' and age:
            print(age)
            women_count+=1
            write_image(filename)
        
        elif gender == 'M' and men_count < women_max_count and age:
            print(age)
            men_count+=1
            write_image_gender(filename)

def age_and_gender_division(start_path):
    # for f in os.listdir(end_path_age_and_gender):
    #     os.remove(f)

    men_count = 0
    women_count = 0
    df_ages = df[df['age'].notna()]
    women_max_count = df_ages['gender'].value_counts().K
    no_age = 0

    for filename in os.listdir(start_path):
        
        if filename in df_ages.values:
            row = df_ages.loc[df_ages['image'] == filename]
            print('ROW: ')
            print(row)
            gender = row['gender'].values[0]
            age = row['age'].values[0]

            if (men_count >= women_max_count) and (women_count >= women_max_count):
                break

            elif np.isnan(age):
                no_age +=1
            
            elif gender == 'K' and age:
                women_count+=1
                write_image_age_and_gender(filename)
        
            elif gender == 'M' and men_count < women_max_count and age:
                men_count+=1
                write_image_age_and_gender(filename)
    
    print('Women count:')
    print(women_count)

    print('Men count: ')
    print(men_count)

    print('Total pictures: ')
    print(women_count+men_count)

def age_division(start_path):
    for f in os.listdir(end_path_age):
        os.remove(f)

    df_ages = df[df['age'].notna()]
    no_age = 0
    for filename in os.listdir(start_path):
        if filename in df_ages.values:
            row = df.loc[df['image'] == filename]
            age = row['age'].values[0]

            if not np.isnan(age):
                write_image_age(filename)

            else:
                no_age +=1
                print('No age!')

age_and_gender_division(start_path)