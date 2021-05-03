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

fldr="resized_equal_distribution_pictures"
df = pd.read_csv('../data/full_dataset.csv', sep=';')
   
face_crop = []
genders = []
ages = []
for filename in os.listdir(fldr):
    print('Processing: '+filename)
    img = cv2.imread(fldr+'/' + filename)
    face_crop.append(img)
    row = df.loc[df['image'] == filename]
    gender = row['gender'].values[0]
    age = row['age'].values[0]

    if gender == 'K':
        genders.append(1)
    elif gender == 'M':
        genders.append(0)
    else:
        genders.append(1)
        print('NO_GENDER_ERROR')

    if not np.isnan(age):
        ages.append(int(age))
    else:
        ages.append(int(40.0))
        print('NO_AGE_ERROR')
face_crop_f = np.array(face_crop)
genders_f = np.array(genders)
ages_f = np.array(ages)

#np.save('face_crop.npy',face_crop_f)
#np.save('genders.npy',genders_f) 
#np.save('ages.npy',ages_f) 

labels=[]
i=0
while i<len(ages):
  label=[]
  label.append([ages[i]])
  label.append([genders[i]])
  labels.append(label)
  i+=1

face_crop_f_2=face_crop_f/255  ##dividing an image by 255 simply rescales the image from 0-255 to 0-1.
labels_f = np.array(labels) 

X_train, X_test, y_train, y_test= train_test_split(face_crop_f_2, labels_f,test_size=0.25)

y_train_2=[y_train[:,1],y_train[:,0]]
y_test_2=[y_test[:,1],y_test[:,0]]

##Getting the already trained model
#Model = keras.models.load_model('Age_sex_detection_full_dataset_equal.h5')

##Train a new model
Model = train_model(X_train, X_test, y_train_2, y_test_2)

print('Evaluating model:')
Model.evaluate(X_test,y_test_2)
print('Predicting:')
pred=Model.predict(X_test)


i=0
pred_gender=[]
pred_age_errors=[]
while(i<len(pred[0])):
    pred_gender.append(int(np.round(pred[0][i])))
    pred_age_errors.append(abs(pred[1][i] - y_test_2[1][i]))
    i+=1


## Visualizations
# fig, ax = plt.subplots()
# ax.scatter(y_test_2[1], pred[1])
# ax.plot([y_test_2[1].min(),y_test_2[1].max()], [y_test_2[1].min(), y_test_2[1].max()], 'k--', lw=4)
# ax.set_xlabel('Actual Age')
# ax.set_ylabel('Predicted Age')
# plt.show()

results = confusion_matrix(y_test_2[0], pred_gender)
print(results)

report_gender=classification_report(y_test_2[0], pred_gender)
print('Report gender: ')
print(report_gender)

mean_age_error = np.mean(pred_age_errors)
print('Age standard error: ' + str(mean_age_error))
#report_age=classification_report(y_test_2[1], pred_age)
#accuracy = accuracy_score(y_test_2[1], pred_age

