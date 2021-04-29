import os
import numpy as np
import cv2
import pandas as pd
from tensorflow import keras
from Model_age_sex_detection import train_model

fldr="gender_cropped_faces"

face_crop = []
for filename in os.listdir(fldr):
    img = cv2.imread(fldr+'/' + filename)
    face_crop.append(img)

face_crop_f = np.array(face_crop)
face_crop_f_2=face_crop_f/255 
    

Model = keras.models.load_model('Age_sex_detection_full_dataset_equal.h5')
pred=Model.predict(face_crop_f_2)
i=0
Pred_l=[]
Pred_age=[]
while(i<len(pred[0])):
    Pred_l.append(int(np.round(pred[0][i])))
    Pred_age.append(int(np.round(pred[1][i])))
    i+=1


females = 0
males = 0
for i in range(len(face_crop_f_2)):
    cv2.imshow('face',face_crop_f_2[i])
    
    print('PRED GENDER: ')
    gender = ""
    
    if Pred_l[i] == 1:
        females +=1
        gender = "Female"
    elif Pred_l[i] == 0:
        males +=1
        gender = "Male"
    print(gender)
    print('PRED AGE: ')
    print(Pred_age[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print('females:')
print(females)
print('males: ')
print(males)
