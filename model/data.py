import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

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

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')

    index = 0
    for image in X_train:
        x = image.reshape((1,) + image.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1):
            i += 1
            X_train = np.concatenate((X_train, batch))
            y_train = np.append(y_train, [y_train[index]], axis=0)
            if i > 4:
                print('done with 1 batch')
                break
        index+=1

    return X_train, X_test, y_train, y_test
