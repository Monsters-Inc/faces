import os
import numpy as np
import cv2
import pandas as pd
##Tensorflow imports
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
##SkLearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


EPOCHS = 500

def model():
    input_shape=(48, 48, 3)
    inputs = Input((input_shape))
    conv_1 = Conv2D(filters=32,
            kernel_size=(3, 3),
            padding = 'same',
            strides=(1, 1),
            kernel_regularizer=l2(0.001),
            activation='relu')(inputs)
    drop_1 = Dropout(0.1)(conv_1)
    maxp_1 = MaxPooling2D(pool_size = (2,2))(drop_1)

    conv_2 = Conv2D(filters=64,
            kernel_size=(3, 3),
            padding = 'same',
            strides=(1, 1),
            kernel_regularizer=l2(0.001),
            activation='relu')(maxp_1)
    drop_2= Dropout(0.1)(conv_2)
    maxp_2 = MaxPooling2D(pool_size = (2,2))(drop_2)

    conv_3 = Conv2D(filters=128,
            kernel_size=(3, 3),
            padding = 'same',
            strides=(1, 1),
            kernel_regularizer=l2(0.001),
            activation='relu')(maxp_2)
    drop_3= Dropout(0.1)(conv_3)
    maxp_3 = MaxPooling2D(pool_size = (2,2))(drop_3)
    

    conv_4 = Conv2D(filters=256,
            kernel_size=(3, 3),
            padding = 'same',
            strides=(1, 1),
            kernel_regularizer=l2(0.001),
            activation='relu')(maxp_3)
    drop_4= Dropout(0.1)(conv_4)
    maxp_4 = MaxPooling2D(pool_size = (2,2))(drop_4)

    flatten= Flatten()(maxp_4)

    dense_1 = Dense(64,activation='relu')(flatten)
    dense_2 = Dense(64,activation='relu')(flatten)
    
    drop_5 = Dropout(0.2)(dense_1)
    drop_6 = Dropout(0.2)(dense_2)
    
    output_1 = Dense(1,activation="sigmoid",name='sex_out')(drop_5) #Gender
    output_2 = Dense(1,activation="relu",name='age_out')(drop_6) #Age
    
    model = Model(inputs=[inputs], outputs=[output_1,output_2])
    model.compile(loss=["binary_crossentropy","mae"], optimizer="Adam", metrics=["accuracy"])
  
    return model

def train_model(X_train, X_test, Y_train_2, Y_test_2):
    Model=model()
    Model.summary()

    fle_s='Model_two_branches_2.h5'
    checkpointer = ModelCheckpoint(fle_s, monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False, mode='auto',save_freq='epoch')
    Early_stop=tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss',restore_best_weights=True),
    callback_list=[checkpointer,Early_stop]

    Model.fit(X_train,Y_train_2,batch_size=64,validation_data=(X_test,Y_test_2),epochs=EPOCHS,callbacks=[callback_list])
    return Model