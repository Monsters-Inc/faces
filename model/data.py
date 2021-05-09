import os
import cv2
import numpy as np
import pandas as pd
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.model_selection import train_test_split
from preprocessing_tools import binarize_gender, images_to_vectors
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def data(dataset, image_folder, img_shape, test_size, logging):
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

    labels = np.array(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    labels_reshaped = labels.reshape(len(labels), 1)
    labels_one_hot = onehot_encoder.fit_transform(labels_reshaped)

    X = np.array(X)
    X = X / 255
    y = np.array(labels_one_hot)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test
