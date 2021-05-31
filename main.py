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
from preprocessing import preprocess


fldr = 'uploads/'
dest_fldr = 'preprocessedUploads/'
wrong_dest_fldr = 'uploadsWithoutFace/'
Model_g = keras.models.load_model('../../model/g_final.h5')
Model_a = keras.models.load_model('../../model/a_final.h5')
preprocessing_methods = ['he']
img_shape = (96, 96, 1)

# Detects faces and preprocesses the images
preprocess(fldr, dest_fldr, wrong_dest_fldr, img_shape, preprocessing_methods)

# Creating array with names of the images where faces couldn't be found.
filenames_no_face = []
for filename in os.listdir(wrong_dest_fldr):
    filenameWithoutSpaces = filename.replace(" ", "")
    filenames_no_face.append(filenameWithoutSpaces)

# Creating list with the preprocessed pictures and list with the corresponding filenames  
pictures = []
filenames = []
for filename in os.listdir(dest_fldr): 
    img = cv2.imread(dest_fldr+'/' + filename, 0)
    pictures.append(img)
    filenameWithoutSpaces = filename.replace(" ", "")
    filenames.append(filenameWithoutSpaces)

pictures_f = np.array(pictures)
pictures_f_2 = pictures_f/255

# Predicts for age and gender

pred_gender = Model_g.predict(pictures_f_2)
pred_age = Model_a.predict(pictures_f_2)
  
res = ""
for i in range(len(pred_gender)):
    
    sex_f = ['Male','Female']
    age = int(pred_age[i])
    sex = int(np.argmax(pred_gender[i]))
    
    final_prediction = [str(age),sex_f[sex]]
    res = res + filenames[i] + " " + final_prediction[0] + " " + final_prediction[1] + " "
 

# Creating the string to return 

res = res + '*' + " "
for i in range(len(filenames_no_face)):
    res = res + filenames_no_face[i] + " "

print(res)

#Removing images
for file in os.listdir('uploads'):

    os.remove(os.path.join('uploads', file))

for file in os.listdir('preprocessedUploads'):
    os.remove(os.path.join('preprocessedUploads', file))

for file in os.listdir('uploadsWithoutFace'):
    os.remove(os.path.join('uploadsWithoutFace', file))



