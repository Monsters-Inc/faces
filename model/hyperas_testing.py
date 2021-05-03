from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D
from tensorflow.keras.layers import Input, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def data():
    fldr="resized_equal_distribution_pictures"
    df = pd.read_csv('../data/full_dataset.csv', sep=';')
    
    face_crop = []
    genders = []
    # ages = []
    for filename in os.listdir(fldr):
        print('Processing: '+filename)
        img = cv2.imread(fldr+'/' + filename)
        face_crop.append(img)
        row = df.loc[df['image'] == filename]
        gender = row['gender'].values[0]
        # age = row['age'].values[0]

        if gender == 'K':
            genders.append(1)
        elif gender == 'M':
            genders.append(0)
        else:
            genders.append(1)
            print('NO_GENDER_ERROR')

        # if not np.isnan(age):
        #     ages.append(int(age))
        # else:
        #     ages.append(int(40.0))
        #     print('NO_AGE_ERROR')

    face_crop_f = np.array(face_crop)
    genders_f = np.array(genders)
    # ages_f = np.array(ages)
    # labels=[]
    # i=0
    # while i<len(ages):
    #     label=[]
    #     label.append([ages[i]])
    #     label.append([genders[i]])
    #     labels.append(label)
    #     i+=1

    face_crop_f_2=face_crop_f/255  ##dividing an image by 255 simply rescales the image from 0-255 to 0-1.
    # labels_f = np.array(labels) 

    x_train, x_test, y_train, y_test= train_test_split(face_crop_f_2, genders_f,test_size=0.25)

    # y_train_2=[y_train[:,1],y_train[:,0]]
    # y_test_2=[y_test[:,1],y_test[:,0]]


    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    input_shape=(48, 48, 3)

    inputs = Input((input_shape))
    conv_1 = Conv2D(filters={{choice([16, 32, 64, 128, 256])}},
            kernel_size = {{choice([(1, 1), (3, 3), (5, 5)])}},
            padding = 'same',
            strides = (1,1),
            kernel_regularizer=l2({{uniform(0.0001, 0.001)}}),
            activation={{choice(['relu', 'sigmoid'])}})(inputs)
    drop_1 = Dropout({{uniform(0, 1)}})(conv_1)
    maxp_1 = MaxPooling2D(pool_size = {{choice([(1, 1), (2, 2), (3, 3), (4, 4)])}})(drop_1)

    conv_2 = Conv2D(filters={{choice([16, 32, 64, 128, 256])}},
            kernel_size={{choice([(1, 1), (3, 3), (5, 5)])}},
            padding = 'same',
            strides=(1,1),
            kernel_regularizer=l2({{uniform(0.0001, 0.001)}}),
            activation={{choice(['relu', 'sigmoid'])}})(maxp_1)
    drop_2= Dropout({{uniform(0, 1)}})(conv_2)
    maxp_2 = MaxPooling2D(pool_size = {{choice([(1, 1), (2, 2), (3, 3), (4, 4)])}})(drop_2)

    conv_3 = Conv2D(filters={{choice([16, 32, 64, 128, 256])}},
            kernel_size={{choice([(1, 1), (3, 3), (5, 5)])}},
            padding = 'same',
            strides=(1,1),
            kernel_regularizer=l2({{uniform(0.0001, 0.001)}}),
            activation={{choice(['relu', 'sigmoid'])}})(maxp_2)
    drop_3= Dropout({{uniform(0, 1)}})(conv_3)
    maxp_3 = MaxPooling2D(pool_size = {{choice([(1, 1), (2, 2), (3, 3), (4, 4)])}})(drop_3)
    

    conv_4 = Conv2D(filters={{choice([16, 32, 64, 128, 256])}},
            kernel_size={{choice([(1, 1), (3, 3), (5, 5)])}},
            padding = 'same',
            strides=(1,1),
            kernel_regularizer=l2({{uniform(0.0001, 0.001)}}),
            activation={{choice(['relu', 'sigmoid'])}})(maxp_3)
    drop_4= Dropout({{uniform(0, 1)}})(conv_4)
    maxp_4 = MaxPooling2D(pool_size = {{choice([(1, 1), (2, 2), (3, 3), (4, 4)])}})(drop_4)

    flatten= Flatten()(maxp_4)

    dense_1 = Dense({{choice([64, 128, 256, 512])}},activation={{choice(['relu', 'sigmoid'])}})(flatten)
    dense_2 = Dense({{choice([64, 128, 256, 512])}},activation={{choice(['relu', 'sigmoid'])}})(flatten)
    
    drop_5 = Dropout({{uniform(0, 1)}})(dense_1)
    #drop_6 = Dropout({{uniform(0, 1)}})(dense_2)
    
    output_1 = Dense(1,activation={{choice(['relu', 'sigmoid'])}},name='sex_out')(drop_5) #Gender
    #output_2 = Dense(1,activation={{choice(['relu', 'sigmoid'])}},name='age_out')(drop_6) #Age
    
    model = Model(inputs=[inputs], outputs=[output_1])
    model.compile(loss=["binary_crossentropy","mae"], optimizer="Adam", metrics=["accuracy"])

    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128, 256])}},
              epochs={{choice([250, 400, 500, 600, 700])}},
              verbose=2,
              validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}



best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=5,
                                        trials=Trials())
x_train, Y_train, x_test, Y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(x_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)