import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing_tools import binarize_gender, images_to_vectors
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# This cannot be used in this current version of the model
def data(dataset, image_folder, img_shape, test_size, logging):

    color = 1 if img_shape[2] == 3 else 0

    # Import dataset
    df = pd.read_csv(dataset, sep=';')

    # Image list from image column in df
    images = df['image'].tolist()

    # Binarize gender
    df['gender'] = binarize_gender(df)

    # Convert images to vectors
    X_data, new_df = images_to_vectors(images, image_folder, df, color, logging)

    # Gender list from grender column in df
    labels = new_df['gender'].values

    y = labels

    # Remove 1 dimension entries from the shape of an array
    X = np.array(np.squeeze(X_data))

    # Normalize images
    X = X.astype('float32')

    # Splitting into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train_reshaped = X_train.reshape(
        X_train.shape[0], img_shape[0], img_shape[1], img_shape[2]) / 255
    X_test_reshaped = X_test.reshape(
        X_test.shape[0], img_shape[0], img_shape[1], img_shape[2]) / 255
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    return X_train_reshaped, X_test_reshaped, y_train_one_hot, y_test_one_hot


def data_final(dataset, image_folder, img_shape, test_size, logging):
    # Depending on how many channels the image is in color or not
    color = 1 if img_shape[2] == 3 else 0

    # Read Dataset
    df = pd.read_csv(dataset, sep=';')

    X = []
    men = []
    women = []

    for filename in os.listdir(image_folder):
        if logging:
            print('Processing: '+filename)
        img = cv2.imread(image_folder+'/' + filename, color)
        X.append(img)
        row = df.loc[df['image'] == filename]
        gender = row['gender'].values[0]

        if gender == 'K' or gender == 'F':
            women.append(1)
            men.append(0)
        elif gender == 'M':
            women.append(0)
            men.append(1)
        else:
            print('NO_GENDER_ERROR')
            quit()

    labels = []
    for i in range(len(men)):
        labels.append([men[i], women[i]])

    X = np.array(X)
    X = X / 255  # Rescales the image from 0-255 to 0-1.
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print(y_train)

    y_train_reformatted = [y_train[:, 1], y_train[:, 0]]
    y_test_reformatted = [y_test[:, 1], y_test[:, 0]]

    return X_train, X_test, y_train_reformatted, y_test_reformatted

def data_final_other(dataset, image_folder, img_shape, test_size, logging):
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

    # label_encoder = LabelEncoder()
    # integer_labels = label_encoder.fit_transform(labels) # Encodes each letter as an integer

    onehot_encoder = OneHotEncoder(sparse=False)
    labels = np.array(labels)
    labels_reshaped = labels.reshape(len(labels), 1)
    labels_one_hot = onehot_encoder.fit_transform(labels_reshaped)

    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    print(labels_one_hot)

    X = np.array(X)
    X = X / 255  # Rescales the image from 0-255 to 0-1.
    y = np.array(labels_one_hot)

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    y_train_reformatted = y_train # [y_train[:, 1], y_train[:, 0]]
    y_test_reformatted = y_test # [y_test[:, 1], y_test[:, 0]]

    #print('\n\n##########Y TRAIN LENGTH: ', str(len(y_train_reformatted)))
    #print('\n\n##########Y TEST LENGTH: ', str(len(y_test_reformatted)))

    return X_train, X_test, y_train_reformatted, y_test_reformatted
