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
from Model_age_sex_detection import train_model
#from model import train_model

fldr="../model/age_gender_RGB"
#df = pd.read_csv('../data/dataset.csv', sep=';')
df = pd.read_csv('../data/full_dataset.csv', sep=';')
   
face_crop = []
genders = []
ages = []
for filename in os.listdir(fldr):
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

np.save('face_crop.npy',face_crop_f)
np.save('genders.npy',genders_f) 
np.save('ages.npy',ages_f) 

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

X_train, X_test, Y_train, Y_test= train_test_split(face_crop_f_2, labels_f,test_size=0.25)

Y_train_2=[Y_train[:,1],Y_train[:,0]]
Y_test_2=[Y_test[:,1],Y_test[:,0]]

##Getting the already trained model
Model = keras.models.load_model('prev_models/Age_sex_detection_full_dataset_equal.h5')

##Train a new model
#Model = train_model(X_train, X_test, Y_train_2, Y_test_2)
Model.evaluate(X_test,Y_test_2)
pred=Model.predict(X_test)

print('PREED: ')
print(pred)


# labels_f_2=[labels_f[:,1],labels_f[:,0]]
# Model.evaluate(face_crop_f_2,labels_f_2)
# pred=Model.predict(face_crop_f_2)



i=0
Pred_l=[]
Pred_age=[]
while(i<len(pred[0])):
    Pred_l.append(int(np.round(pred[0][i])))
    Pred_age.append(int(np.round(pred[1][i])))
    i+=1

##For checking which images the algorithm guessed incorrectly
# Y_test_genders = Y_test_2[0]
# Wrong_guesses_indexes = []
# for i in range(len(Pred_l)):
#     if Pred_l[i] != Y_test_genders[i][0]:
#         Wrong_guesses_indexes.append(i)


# for i in Wrong_guesses_indexes:
#     cv2.imshow('face',X_test[i])
#     print('PRED: ')
#     print(Pred_l[i])
#     print('Y-TEST: ')
#     print(Y_test_genders[i][0])
#     cv2.waitKey(0)
    #cv2.destroyAllWindows()

fig, ax = plt.subplots()
ax.scatter(Y_test_2[1], pred[1])
ax.plot([Y_test_2[1].min(),Y_test_2[1].max()], [Y_test_2[1].min(), Y_test_2[1].max()], 'k--', lw=4)
ax.set_xlabel('Actual Age')
ax.set_ylabel('Predicted Age')
plt.show()

# fig, ax = plt.subplots()
# ax.scatter(labels_f_2[1], pred[1])
# ax.plot([labels_f_2[1].min(),labels_f_2[1].max()], [labels_f_2[1].min(), labels_f_2[1].max()], 'k--', lw=4)
# ax.set_xlabel('Actual Age')
# ax.set_ylabel('Predicted Age')
# plt.show()

results = confusion_matrix(Y_test_2[0], Pred_l)
print(results)

# results = confusion_matrix(labels_f_2[0], Pred_l)
# print(results)

report=classification_report(Y_test_2[0], Pred_l)
report2=classification_report(Y_test_2[1], Pred_age)
accuracy = accuracy_score(Y_test_2[1], Pred_age)
print(report)

# print('REPORT GENDER: ')
# print(report)
# print('ACCURACY AGE: ')
# print(accuracy)


# def test_image(ind,images_f,images_f_2,Model):  
#     #cv2_imshow(images_f[ind])
#     cv2.imshow('face', images_f[ind])
#     cv2.waitKey(0)
#     image_test=images_f_2[ind]
#     pred_1=Model.predict(np.array([image_test]))
#     #print(pred_1)
#     sex_f=['Male','Female']
#     age=int(np.round(pred_1[1][0]))
#     sex=int(np.round(pred_1[0][0]))
#     print("Predicted Age: "+ str(age))
#     print("Predicted Sex: "+ sex_f[sex])

# for i in Wrong_guesses_indexes:
#     test_image(i,face_crop_f,face_crop_f_2,Model)
