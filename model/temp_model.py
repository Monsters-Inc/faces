#coding=utf-8

from __future__ import print_function

try:
    import numpy as np
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    import os
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    from tensorflow import keras
except:
    pass

try:
    from tensorflow.keras.layers import Dropout
except:
    pass

try:
    from tensorflow.keras.layers import Flatten, BatchNormalization
except:
    pass

try:
    from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D
except:
    pass

try:
    from tensorflow.keras.layers import Input, Activation, Add
except:
    pass

try:
    from tensorflow.keras.models import Model
except:
    pass

try:
    from tensorflow.keras.regularizers import l2
except:
    pass

try:
    from tensorflow.keras.optimizers import Adam
except:
    pass

try:
    from tensorflow.keras.callbacks import ModelCheckpoint
except:
    pass

try:
    import cv2
except:
    pass

try:
    from sklearn.model_selection import train_test_split
except:
    pass

try:
    from sklearn.metrics import log_loss
except:
    pass

try:
    from sklearn.preprocessing import LabelEncoder
except:
    pass

try:
    from sklearn.preprocessing import OneHotEncoder
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

dataset = "../data/full_dataset.csv"
image_folder = "resized_96_equal_distribution_pictures/"
img_shape = (96, 96, 3)
test_size = 0.25
logging = False


# Depending on how many channels the image is in color or not
color = 1 if img_shape[2] == 3 else 0

# Read Dataset
df = pd.read_csv(dataset, sep=';')

X = []
labels = []
for filename in os.listdir(image_folder):
    if logging:
        print('Processing: '+filename)
    img = cv2.imread(image_folder+'/' + filename, color)
    X.append(img)
    row = df.loc[df['image'] == filename]
    gender = row['gender'].values[0]

    if gender == 'K' or gender == 'F':
        labels.append(1)
    elif gender == 'M':
        labels.append(0)
    else:
        print('NO_GENDER_ERROR')
        quit()

labels = labels
onehot_encoder = OneHotEncoder(sparse=False)
labels_reshaped = labels.reshape(len(labels), 1)
labels_one_hot = onehot_encoder.fit_transform(labels_reshaped)

X = np.array(X)
X = X / 255
y = labels_one_hot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)



def keras_fmin_fnct(space):

    inputs = Input((96,96,3))

    x = Conv2D(filters=space['filters'], kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(0.00015))(input)
    x = Dropout(space['Dropout'])(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(filters=space['filters_1'], kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(0.00015))(input)
    x = Dropout(space['Dropout_1'])(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(filters=space['filters_2'], kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(0.00015))(input)
    x = Dropout(space['Dropout_2'])(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(filters=space['filters_3'], kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(0.00015))(input)
    x = Dropout(space['Dropout_3'])(x)
    x = Conv2D(filters=space['filters_4'], kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(0.00015))(input)
    x = Dropout(space['Dropout_4'])(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(filters=space['filters_5'], kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(0.00015))(input)
    x = Dropout(space['Dropout_5'])(x)
    x = Conv2D(filters=space['filters_6'], kernel_size=(3, 3), activation='relu', padding='same', strides=(1, 1), kernel_regularizer=l2(0.00015))(input)
    x = Dropout(space['Dropout_6'])(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Flatten()(x)

    x = Dense(64, activation='relu')(x)

    output = Dense(2, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=["binary_crossentropy", "binary_crossentropy"], optimizer="Adam", metrics=["accuracy"])

    result = model.fit(x_train, y_train,
              batch_size=space['batch_size'],
              epochs=space['epochs'],
              verbose=2,
              validation_split=0.1)

    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_accuracy']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'filters': hp.choice('filters', [16, 32, 64, 128, 256, 512]),
        'Dropout': hp.uniform('Dropout', 0, 1),
        'filters_1': hp.choice('filters_1', [16, 32, 64, 128, 256, 512]),
        'Dropout_1': hp.uniform('Dropout_1', 0, 1),
        'filters_2': hp.choice('filters_2', [16, 32, 64, 128, 256, 512]),
        'Dropout_2': hp.uniform('Dropout_2', 0, 1),
        'filters_3': hp.choice('filters_3', [16, 32, 64, 128, 256, 512]),
        'Dropout_3': hp.uniform('Dropout_3', 0, 1),
        'filters_4': hp.choice('filters_4', [16, 32, 64, 128, 256, 512]),
        'Dropout_4': hp.uniform('Dropout_4', 0, 1),
        'filters_5': hp.choice('filters_5', [16, 32, 64, 128, 256, 512]),
        'Dropout_5': hp.uniform('Dropout_5', 0, 1),
        'filters_6': hp.choice('filters_6', [16, 32, 64, 128, 256, 512]),
        'Dropout_6': hp.uniform('Dropout_6', 0, 1),
        'batch_size': hp.choice('batch_size', [64, 128, 256]),
        'epochs': hp.choice('epochs', [250, 400, 500, 600, 700]),
    }
