import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sys

# fldr = 'webb/server/uploads/'
# dest_fldr = 'webb/server/preprocessedUploads/'
# wrong_dest_fldr = 'webb/server/uploadsWithoutFace/'
# Model = keras.models.load_model('Liv_Ullis_Testing/prev_models/Age_sex_detection_full_dataset_equal.h5')
fldr = 'uploads/'
dest_fldr = 'preprocessedUploads/'
wrong_dest_fldr = 'uploadsWithoutFace/'
Model = keras.models.load_model('../../Liv_Ullis_Testing/prev_models/Age_sex_detection_full_dataset_equal.h5')

# Cropping, resizing and filtering the images and puts them in destination_fldr (DENNA FUNKTION SKA BYTAS MOT DEN NYA FUNKTIONEN)
def preprocessing(start_path, end_path, wrong_path):
    ##Import cascade files for the classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    alt_tree = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

    wrongs = 0
    ##Try the different haarcascade files to see which works for the picture
    for filename in os.listdir(start_path):
        img = cv2.imread(start_path+'/' + filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## cv2.COLOR_BGR2RGB ? 
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

        if len(faces) == 0:
            faces = face_cascade_profile.detectMultiScale(gray_img, 1.3, 5)
        if len(faces) == 0:
            flipped = cv2.flip(gray_img, 1)
            faces = face_cascade_profile.detectMultiScale(flipped, 1.3, 5)
        if len(faces) == 0:
            faces = alt.detectMultiScale(gray_img, 1.3, 5)
        if len(faces) == 0:
            faces = alt2.detectMultiScale(gray_img, 1.3, 5)
        if len(faces) == 0:
            faces = alt_tree.detectMultiScale(gray_img, 1.3, 5)

        ## We want the face with the largest area
        sqr_area = 0
        index = 0
        biggest_area_index = 0 ## index of face with biggest area
        for (x,y,w,h) in faces:
            current_area = w*h
            if sqr_area < current_area:
                biggest_area_index = index
                sqr_area = current_area
            index+=1
        if (len(faces) != 0):      
            x = faces[biggest_area_index][0]
            y = faces[biggest_area_index][1]
            w = faces[biggest_area_index][2]
            h = faces[biggest_area_index][3]
            img = cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),2)

            img_cropped = img[y:y+h, x:x+w]
            img_resized = cv2.resize(img_cropped,(48,48))
            
            cv2.imwrite(end_path+filename, img_resized)

        else:
            cv2.imwrite(wrong_path+filename, img)
            wrongs+=1

preprocessing(fldr, dest_fldr, wrong_dest_fldr)


# Creating array with names of the images where faces couldn't be found.
filenames_no_face = []
for filename in os.listdir(wrong_dest_fldr):
    filenameWithoutSpaces = filename.replace(" ", "")
    filenames_no_face.append(filenameWithoutSpaces)

# Creating list with the preprocessed pictures and list with the corresponding filenames  
pictures = []
filenames = []
for filename in os.listdir(dest_fldr):
    img = cv2.imread(dest_fldr+'/' + filename)
    pictures.append(img)
    filenameWithoutSpaces = filename.replace(" ", "")
    filenames.append(filenameWithoutSpaces)

pictures_f = np.array(pictures)
pictures_f_2=pictures_f/255

## DENNA SKA BYTAS UT MOT TVÅ NYA FÄRDIGTRÄNADE MODELLER, EN FÖR AGE OCH EN FÖR GENDER
pred=Model.predict(pictures_f_2)

def test_image(ind,images_f,images_f_2,Model):  
    image_test=images_f_2[ind]
    pred_1=Model.predict(np.array([image_test]))
    sex_f=['Male','Female']
    age=int(np.round(pred_1[1][0]))
    sex=int(np.round(pred_1[0][0]))
    return_value = [str(age),sex_f[sex]]
    return(return_value)
 
# Creating the string to return 
res = ""
for  i in range(len(pictures)):
    final_prediction = test_image(i,pictures_f,pictures_f_2,Model)
    res = res + filenames[i] + " " + final_prediction[0] + " " + final_prediction[1] + " "

res = res + '*' + " "
for  i in range(len(filenames_no_face)):
    res = res + filenames_no_face[i] + " "

print(res)

#Removing images
for f in os.listdir('uploads'):
    os.remove(os.path.join('uploads', f))

for f in os.listdir('preprocessedUploads'):
    os.remove(os.path.join('preprocessedUploads', f))

for f in os.listdir('uploadsWithoutFace'):
    os.remove(os.path.join('uploadsWithoutFace', f))



