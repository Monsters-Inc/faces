import os
import cv2
import numpy as np
import pandas as pd
#import dlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessing_tools import format_folder_name, he_single, clahe_single, canny_edges_single, median_filtering_single
import random

face_detection_model = 'premade.dat'

def get_images(paths, dir_path, image_shape, preprocessing, logging):
    dir_path = format_folder_name(dir_path)
    images = []
    count = 0
    for image in paths:
        if logging:
            print('Getting Images: '+image+' | ' +
                  str(count)+'/'+str(len(paths)), end='\r')
        temp = cv2.imread(dir_path + image)
        temp = cv2.resize(temp, (image_shape[0], image_shape[1]))
        temp = preprocess([temp], image_shape, preprocessing)

        images.append(temp[0])
        count += 1

    return np.asarray(images)


def data_age(dataset, image_folder, EQUAL, image_shape, test_size, preprocessing, logging):
    df = pd.read_csv(dataset, sep=';')
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
        df.image.values, df['age'].values, test_size=test_size)

    train_images = get_images(X_train, image_folder, image_shape, preprocessing, logging)
    test_images = get_images(X_test, image_folder, image_shape, preprocessing, logging)

    return train_images, test_images, y_train, y_test

def augumentation(X_train):
  datagen = ImageDataGenerator(
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      #shear_range=0.2,
      #zoom_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest'
  )

  datagen.fit(X_train)

  return datagen


def data_gender(dataset, image_folder, img_shape, test_size, preprocessing, logging):
    image_folder = format_folder_name(image_folder)
    # Depending on how many channels the image is in color or not
    color = 1 if img_shape[2] == 3 else 0

    # Read Dataset
    df = pd.read_csv(dataset, sep=';')

    folder = os.listdir(image_folder)
    folder = random.sample(folder, len(folder))
    X=[]
    labels=[]
    for filename in folder:
        if logging:
            print('Processing: '+filename)
        img=cv2.imread(image_folder+'/' + filename, color)
        
        img = preprocess([img], img_shape, preprocessing)
        img = img[0]

        X.append(img)
        row=df.loc[df['image'] == filename]
        gender=row['gender'].values[0]

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

    X = np.asarray(X)
    #X = X / 255.0
    y = np.array(labels_one_hot)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


# def get_faces(image):

#   model=dlib.cnn_face_detection_model_v1(face_detection_model)
#   faces=model(image)

#   cropped=[]

#   for face in faces:
#     x=face.rect.left()
#     y=face.rect.top()
#     w=face.rect.right()
#     h=face.rect.bottom()
#     x=max(0, x)
#     y=max(0, y)
#     w=max(0, w)
#     h=max(0, h)

#     crop=image[y:h, x:w]
#     cropped.append(crop)
#     # cv2.imshow('bild', crop)
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()

#   return cropped

def preprocess(input_data, img_shape, preprocessing):
  faces=[]
  if isinstance(input_data, str):
    input_data=format_folder_name(input_data)
    folder=sorted(os.listdir(input_data))

    images = []
    for file in folder:
      img=cv2.imread(input_data + file)
      images.append(img)


    faces=images

    # for image in images:
    #  cropped_faces=get_faces(image)
    #  faces=faces + cropped_faces

  else:
    faces=input_data

  for i in range(len(faces)):
    faces[i]=cv2.resize(faces[i], (img_shape[0], img_shape[1]))

    if len(preprocessing) > 0:
      if 'gray' in preprocessing:
        if img_shape[2] == 3:
          faces[i]=cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
      if 'he' in preprocessing:
        faces[i]=he_single(faces[i])

      if 'clahe' in preprocessing:
        faces[i]=clahe_single(faces[i])

      if 'canny' in preprocessing:
        faces[i]=canny_edges_single(faces[i])
      
      if 'median' in preprocessing:
        faces[i]=median_filtering_single(faces[i])

    faces[i]=faces[i] / 255.0


  faces=np.asarray(faces)

  return faces

