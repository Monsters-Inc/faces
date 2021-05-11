import pandas as pd
import tensorflow as tf
import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Dense, MaxPooling2D, Conv2D, Input, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3

##SkLearn imports
from sklearn.model_selection import train_test_split

EQUAL = False
img_dim = 48 
img_path = '../dataset/'

def get_model():
    input_shape = (img_dim, img_dim, 1)
    inputs = Input((input_shape))
    conv_1 = Conv2D(filters=32,
                    kernel_size=(3, 3),
                    padding='same',
                    strides=(1, 1),
                    kernel_regularizer=l2(0.001),
                    activation='relu')(inputs)
    maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    conv_2 = Conv2D(filters=64,
                    kernel_size=(3, 3),
                    padding='same',
                    strides=(1, 1),
                    kernel_regularizer=l2(0.001),
                    activation='relu')(maxp_1)
    maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = Conv2D(filters=128,
                    kernel_size=(3, 3),
                    padding='same',
                    strides=(1, 1),
                    kernel_regularizer=l2(0.001),
                    activation='relu')(maxp_2)
    maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    conv_4 = Conv2D(filters=256,
                    kernel_size=(3, 3),
                    padding='same',
                    strides=(1, 1),
                    kernel_regularizer=l2(0.001),
                    activation='relu')(maxp_3)
    maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    flatten = Flatten()(maxp_4)
    #drop_f = Dropout(0.8)(flatten)

    dense_1 = Dense(64, activation='relu')(flatten)
    drop_5 = Dropout(0.2)(dense_1)
    dense_2 = Dense(64, activation='relu')(drop_5)
    drop_6 = Dropout(0.2)(drop_5)

    output_1 = Dense(1, activation="relu")(drop_6)

    model = Model(inputs=[inputs], outputs=[output_1])
    model.compile(loss=["mae"],
                  optimizer="adam", metrics=["mean_absolute_error"])

    return model

def get_images(paths, dir_path, flip = False):
  images = []
  for image in paths:
    temp = cv.imread(dir_path + image)
    temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    temp = cv.resize(temp, (img_dim, img_dim))

    if flip:
      flipped = cv.flip(temp, 1)
      flipped = flipped[..., np.newaxis]
      images.append(flipped)

    temp = temp[..., np.newaxis]
    images.append(temp)

  return np.asarray(images)

def train_model():

  model = get_model()

  df = pd.read_csv('full_dataset.csv', sep=';')
  df.dropna(0, 'any', inplace=True)

  if EQUAL:
    women = df.loc[df['gender'] == 'K', ['image', 'age']]
    sample_size = len(women.index)

    men = df.loc[df['gender'] == 'M', ['image', 'age']]

    women = women.sample(n=sample_size)
    men = men.sample(n=sample_size)

    df = women.append(men)

  df = df.sample(frac=1)
  X_train, X_test, y_train, y_test = train_test_split(
      df.image.values, df['age'].values, test_size=0.33)

  train_images = get_images(X_train, img_path, True)
  train_images = train_images / 255.0
  test_images = get_images(X_test, img_path)
  test_images = test_images / 255.0

  y_train = np.repeat(y_train, 2)

  callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

  history = model.fit(train_images, y_train, epochs=500, batch_size=120,
                      validation_data=(test_images, y_test), callbacks=[callback])

  test_loss, test_acc = model.evaluate(test_images,  y_test)

  images = ['1.jpg', '2.jpg', '3.jpg']
  images = get_images(images, '')
  images = images / 255.0

  pred = model.predict(images)
  print(pred)
  

train_model()
