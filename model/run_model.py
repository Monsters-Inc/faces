import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from model import train_model

fldr="resized_96_equal_distribution_pictures/"
df = pd.read_csv('../data/full_dataset.csv', sep=';')

face_crop = []
#genders = []
ages = []
men = []
women = []

for filename in os.listdir(fldr):
    print('Processing: '+filename)
    img = cv2.imread(fldr+'/' + filename)
    face_crop.append(img)
    row = df.loc[df['image'] == filename]
    gender = row['gender'].values[0]
    age = row['age'].values[0]

    if gender == 'K':
        women.append(1)
        men.append(0)
    elif gender == 'M':
        women.append(0)
        men.append(1)
    else:
        print('NO_GENDER_ERROR')
        quit()

    # if not np.isnan(age):
    #     ages.append(int(age))
    # else:
    #     ages.append(int(40.0))
    #     print('NO_AGE_ERROR')
face_crop_f = np.array(face_crop)

labels=[]
i=0
while i<len(men):
  label=[]
  #label.append([ages[i]])
  label.append([men[i]])
  label.append([women[i]])
  labels.append(label)
  i+=1

face_crop_f_2=face_crop_f/255  ##dividing an image by 255 simply rescales the image from 0-255 to 0-1.
labels_f = np.array(labels) 

X_train, X_test, y_train, y_test= train_test_split(face_crop_f_2, labels_f,test_size=0.25)

y_train_2=[y_train[:,1],y_train[:,0]] #y_train[:,2],
y_test_2=[y_test[:,1],y_test[:,0]] #y_test[:,2],

##Getting the already trained model
#Model = keras.models.load_model('Age_sex_detection_full_dataset_equal.h5')

##Train a new model
Model = train_model(X_train, X_test, y_train_2, y_test_2)

print('Evaluating model:')
Model.evaluate(X_test,y_test_2)
print('Predicting:')
pred=Model.predict(X_test)

print(pred)

i=0

pred_women = []
pred_men = []
#pred_age_errors=[]
while(i<len(pred[0])):
    pred_women.append(int(np.round(pred[0][i])))
    pred_men.append(int(np.round(pred[1][i])))
   # pred_age_errors.append(abs(pred[2][i] - y_test_2[2][i]))
    i+=1

results_men = confusion_matrix(y_test_2[1], pred_men)
print(results_men)

results_women = confusion_matrix(y_test_2[0], pred_women)
print(results_women)

report_men=classification_report(y_test_2[1], pred_men)
print('Report men: ')
print(report_men)

report_women=classification_report(y_test_2[0], pred_women)
print('Report women: ')
print(report_women)

# mean_age_error = np.mean(pred_age_errors)
# print('Age standard error: ' + str(mean_age_error))
# #report_age=classification_report(y_test_2[1], pred_age)
# #accuracy = accuracy_score(y_test_2[1], pred_age

