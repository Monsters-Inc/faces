import os
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import log_loss
from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D
from tensorflow.keras.layers import Input, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split


def Convolution(input_tensor, filters):

    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',
               strides=(1, 1), kernel_regularizer=l2(0.00015))(input_tensor)
    x = Dropout(0.1)(x)
    x = Activation('relu')(x)

    return x


def model():
    inputs = Input((96,96,3))

    conv_0 = Convolution(inputs, 16)  # la till 16
    maxp_0 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_0)
    conv_1 = Convolution(maxp_0, 32)
    maxp_1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_1)
    conv_2 = Convolution(maxp_1, 64)
    maxp_2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_2)
    conv_3 = Convolution(maxp_2, 128)
    maxp_3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_3)
    conv_4 = Convolution(maxp_3, 256)
    maxp_4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_4)
    conv_5 = Convolution(maxp_4, 512)
    maxp_5 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_5)
    flatten = Flatten()(maxp_5)

    dense_1 = Dense(64, activation='relu')(flatten)
    dense_2 = Dense(64, activation='relu')(flatten)
    #dense_3 = Dense(64,activation='relu')(flatten)

    drop_1 = Dropout(0.2)(dense_1)
    drop_2 = Dropout(0.2)(dense_2)
    #drop_3 = Dropout(0.2)(dense_3)

    output_1 = Dense(1, activation="sigmoid", name='women_out')(drop_1)  # Women
    output_2 = Dense(1, activation="sigmoid", name='men_out')(drop_2)  # Men
    # output_3 = Dense(1,activation="relu",name='age_out')(drop_3) #Age

    model = Model(inputs=[inputs], outputs=[output_1, output_2])  # , output_3
    model.compile(loss=["binary_crossentropy", "binary_crossentropy"], optimizer="Adam", metrics=["accuracy"])  # ,"mae"

    return model


def train_model(X_train, X_test, Y_train_2, Y_test_2):

    Model = model()
    Model.summary()

    fle_s = 'first_onehot.h5'
    checkpointer = ModelCheckpoint(fle_s, monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    Early_stop = tf.keras.callbacks.EarlyStopping(
        patience=25, monitor='val_loss', restore_best_weights=True),
    callback_list = [checkpointer, Early_stop]

    Model.fit(X_train, Y_train_2, batch_size=64, validation_data=(
        X_test, Y_test_2), epochs=500, callbacks=[callback_list])
    return Model
