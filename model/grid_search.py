from time import time
from model import model as create_model
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
import cv2

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

face_crop_f_2=face_crop_f/255  ##dividing an image by 255 simply rescales the image from 0-255 to 0-1.
#labels_f = np.array(labels) 

X_train, X_test, Y_train, Y_test= train_test_split(face_crop_f_2, genders_f,test_size=0.25)

# Y_train_2=[Y_train[:,1],Y_train[:,0]]


start=time()
model = KerasClassifier(build_fn=create_model)
epochs = np.array([50, 100, 150])
batches = np.array([5, 10, 20])
param_grid = dict(nb_epoch=epochs, batch_size=batches)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.cv_results_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
print("total time:",time()-start)